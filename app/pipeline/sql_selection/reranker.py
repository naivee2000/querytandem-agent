"""Rank revised SQLite candidates without another model request."""

from contextlib import closing
from dataclasses import dataclass
import json
import math
from pathlib import Path
import re
import sqlite3
import time


FEATURES = (
    "agreement", "log_rows", "log_columns", "empty", "log_sql_length",
    "question_overlap", "evidence_overlap", "count_cue", "group_cue",
    "ordering_cue", "distinct_cue", "average_cue", "negation_cue",
)
_WORDS = re.compile(r"[a-z]+|\d+")
_CUES = (
    (r"\b(how many|number of|count)\b", r"\bcount\s*\("),
    (r"\b(each|per|by category)\b", r"\bgroup\s+by\b"),
    (r"\b(top|bottom|highest|lowest|latest|earliest|most|least|first|last)\b",
     r"\border\s+by\b|\b(min|max)\s*\("),
    (r"\b(unique|distinct|different)\b", r"\bdistinct\b|\bgroup\s+by\b"),
    (r"\b(average|mean)\b", r"\bavg\s*\("),
    (r"\b(not|never|without|exclude)\b", r"\b(not|except)\b|!=|<>"),
)


@dataclass(frozen=True)
class Answer:
    rows: frozenset
    columns: int
    row_count: int


@dataclass(frozen=True)
class Group:
    sqls: tuple[str, ...]
    answer: Answer
    features: tuple[float, ...]


def _authorize(action, _name, _column, _database, _source):
    if action in (sqlite3.SQLITE_SELECT, sqlite3.SQLITE_READ, sqlite3.SQLITE_FUNCTION, sqlite3.SQLITE_RECURSIVE):
        return sqlite3.SQLITE_OK
    return sqlite3.SQLITE_DENY


def read_answer(database_path, sql, deadline, *, diagnostics=None):
    """Use the benchmark's unordered result-set comparison semantics."""
    started = time.monotonic()

    def finish(status, answer=None):
        if diagnostics is not None:
            diagnostics.update(status=status, seconds=time.monotonic() - started)
        return answer

    if not sql:
        return finish("missing_sql")
    if started >= deadline:
        return finish("timeout")
    try:
        uri = Path(database_path).resolve().as_uri() + "?mode=ro"
        with closing(sqlite3.connect(uri, uri=True, timeout=max(0, deadline - time.monotonic()))) as conn:
            conn.execute("PRAGMA query_only = ON")
            conn.set_authorizer(_authorize)
            conn.set_progress_handler(lambda: int(time.monotonic() >= deadline), 1000)
            cursor = conn.execute(sql)
            rows = cursor.fetchmany(100_001)
            if time.monotonic() >= deadline:
                return finish("timeout")
            if len(rows) > 100_000:
                return finish("row_limit")
            return finish("ok", Answer(frozenset(rows), len(cursor.description), len(rows)))
    except (sqlite3.Error, ValueError, OSError, TypeError):
        return finish("timeout" if time.monotonic() >= deadline else "sql_error")


def _features(question, evidence, sql, answer, agreement):
    sql = sql.casefold()
    sql_terms = set(_WORDS.findall(sql))
    overlaps = []
    for text in (question, evidence):
        terms = set(_WORDS.findall(text.casefold()))
        overlaps.append(len(terms & sql_terms) / max(1, len(terms)))
    cues = [
        float(bool(re.search(sql_pattern, sql))) if re.search(question_pattern, question.casefold()) else 0.
        for question_pattern, sql_pattern in _CUES
    ]
    return (agreement, math.log1p(answer.row_count), math.log1p(answer.columns),
            float(not answer.rows), math.log1p(len(sql_terms)), *overlaps, *cues)


def group_answers(item, timeout=3.0, *, per_query_timeout=False, diagnostics=None):
    """Use a shared runtime deadline, or independent offline labeling deadlines.

    Runtime callers retain the all-or-nothing timeout fallback. Offline callers
    can retain completed results when a different candidate fails or times out.
    """
    deadline = time.monotonic() + timeout
    executed = {}
    groups = {}
    valid_count = 0
    for sql in item.sql_candidates_after_revision or ():
        if sql not in executed:
            detail = {} if diagnostics is not None else None
            query_deadline = time.monotonic() + timeout if per_query_timeout else deadline
            executed[sql] = read_answer(item.database_path, sql, query_deadline, diagnostics=detail)
            if diagnostics is not None:
                diagnostics.append(dict(sql=sql, **detail))
        answer = executed[sql]
        if answer is not None:
            groups.setdefault(answer.rows, []).append(sql)
            valid_count += 1
    if not per_query_timeout and time.monotonic() >= deadline:
        return []
    results = []
    for sqls in groups.values():
        vectors = [
            _features(item.question, item.evidence or "", sql, executed[sql], len(sqls) / valid_count)
            for sql in dict.fromkeys(sqls)
        ]
        features = tuple(sum(values) / len(values) for values in zip(*vectors))
        results.append(Group(tuple(sqls), executed[sqls[0]], features))
    return results


def majority_vote(groups, current):
    if not groups:
        return current
    most_votes = max(len(group.sqls) for group in groups)
    winners = [group for group in groups if len(group.sqls) == most_votes]
    return winners[0].sqls[0] if len(winners) == 1 and current not in winners[0].sqls else current


@dataclass(frozen=True)
class SQLReranker:
    weights: tuple[float, ...]
    threshold: float

    @classmethod
    def load(cls, path):
        model = json.loads(Path(path).read_text())
        weights = tuple(float(weight) for weight in model["weights"])
        threshold = float(model["threshold"])
        if (model["version"] != 1 or tuple(model["features"]) != FEATURES
                or len(weights) != len(FEATURES) or not all(map(math.isfinite, weights))
                or not 0.5 < threshold < 1):
            raise ValueError("Incompatible SQL reranker artifact.")
        return cls(weights, threshold)

    def choose(self, groups, current):
        incumbent = next((group for group in groups if current in group.sqls), None)
        if incumbent is None or len(groups) < 2:
            return current

        def score(group):
            return math.fsum(weight * feature for weight, feature in zip(self.weights, group.features))

        alternative = max(groups, key=score)
        margin = score(alternative) - score(incumbent)
        if math.isfinite(margin) and margin >= math.log(self.threshold / (1 - self.threshold)):
            return alternative.sqls[0]
        return current
