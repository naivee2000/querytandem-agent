"""Conservative acceptance rules for the first SQLite candidate."""
from contextlib import closing
from pathlib import Path
import re
import sqlite3
import time


_SQL_TOKEN = re.compile(
    r"""--[^\n]*|/\*[\s\S]*?\*/|'(?:''|[^'])*'|"(?:""|[^"])*"|"""
    r"""`(?:``|[^`])*`|\[[^\]]*\]|[a-zA-Z_][\w$]*|[^\s]"""
)
_COMPLEX_QUESTION = re.compile(
    r"\b(?:rank(?:ed|ing)?|top|bottom|highest|lowest|most|least|first|last|"
    r"largest|smallest|best|worst|not|never|without|except|exclude[ds]?|"
    r"compare[ds]?|versus|vs|each|per|difference|ratio|percent(?:age)?|"
    r"more than|less than|higher than|lower than)\b",
    re.IGNORECASE,
)


def accepts_sql(question, sql, database_path, policy, *, timeout=2.0):
    if policy not in ("single_table", "one_join") or not sql or not database_path:
        return False
    if _COMPLEX_QUESTION.search(question):
        return False
    tokens = [token.upper() for token in _SQL_TOKEN.findall(sql)
              if not token.startswith(("--", "/*"))]
    if tokens and tokens[-1] == ";":
        tokens.pop()
    if (not tokens or tokens[0] != "SELECT" or tokens.count("SELECT") != 1
            or tokens.count("FROM") != 1
            or set(tokens).intersection({
                ";", "GROUP", "HAVING", "UNION", "INTERSECT", "EXCEPT", "OVER",
                "LIMIT", "NOT", "NATURAL", "CROSS", "WINDOW",
            })):
        return False
    sources = tokens[tokens.index("FROM") + 1:]
    if "," in sources or "(" in sources[:1]:
        return False
    join_count = tokens.count("JOIN")
    if join_count > (1 if policy == "one_join" else 0):
        return False
    if join_count and "ON" not in sources:
        return False

    deadline = time.monotonic() + timeout
    tables = set()

    def authorize(action, table, column, database, source):
        if source is not None:
            return sqlite3.SQLITE_DENY
        if action == sqlite3.SQLITE_READ:
            if (database != "main" and not (database is None and column == "")
                    or not table or table.lower().startswith("sqlite_")):
                return sqlite3.SQLITE_DENY
            tables.add(table)
        elif action == sqlite3.SQLITE_FUNCTION:
            if column.lower() == "load_extension":
                return sqlite3.SQLITE_DENY
        elif action != sqlite3.SQLITE_SELECT:
            return sqlite3.SQLITE_DENY
        return sqlite3.SQLITE_OK

    try:
        uri = Path(database_path).resolve().as_uri() + "?mode=ro"
        with closing(sqlite3.connect(uri, uri=True, timeout=timeout)) as connection:
            connection.execute("PRAGMA query_only = ON")
            if hasattr(sqlite3, "SQLITE_DBCONFIG_DQS_DML"):
                connection.setconfig(sqlite3.SQLITE_DBCONFIG_DQS_DML, False)
            connection.set_authorizer(authorize)
            connection.set_progress_handler(lambda: time.monotonic() >= deadline, 100)
            rows = connection.execute(sql).fetchmany(100_001)
            return bool(tables) and len(tables) <= join_count + 1 and len(rows) <= 100_000 and time.monotonic() < deadline
    except (OSError, sqlite3.Error, ValueError):
        return False
