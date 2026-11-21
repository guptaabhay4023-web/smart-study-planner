"""
Smart Study Plan - Data Cleaner + Study/Performance Reports (terminal tool)

SETUP (VS Code):  put the messy csv and this file in the same folder -> open Terminal (Ctrl + `) ->
                  pip install pandas numpy
                  python smart_study.py     (clean_study_data.csv is created automatically on every run)

ONE STUDENT'S PERFORMANCE
    python smart_study.py STU0104 2026-12-18     # student + date
    python smart_study.py STU0104                # all days for that student

STUDY HOURS / REPORTS (weekday, month, year ...)
    python smart_study.py --weekday monday               # how much study happened on Mondays (all students)
    python smart_study.py --weekday monday STU0104       # only that student
    python smart_study.py --month dec                    # the whole of December
    python smart_study.py --month 2026-12 --by weekday   # break the month down by weekday
    python smart_study.py --year 2026                    # the whole year
    python smart_study.py --from 2026-12-07 --to 2026-12-11
    python smart_study.py --by weekday | month | year | date | subject    # grouped table
    python smart_study.py --weekday monday --subject physics
    python smart_study.py --month nov --on entry         # use entry date instead of exam date

OTHER
    python smart_study.py --date 2026-12-11     # summary of all students on that day
    python smart_study.py --dates               # which dates exist in the data
    python smart_study.py --clean               # cleaning only -> clean_study_data.csv
    python smart_study.py                       # interactive mode
"""
import argparse
import calendar
import re
import shlex
import sys
from datetime import datetime
from pathlib import Path

try:
    import numpy as np
    import pandas as pd
except ImportError:
    sys.exit("pandas/numpy are not installed. Run this in the terminal:  pip install pandas numpy")

try:                                    # also works in VS Code 'Run' / interactive mode where __file__ is missing
    SCRIPT_DIR = Path(__file__).resolve().parent
except NameError:
    SCRIPT_DIR = Path.cwd()

# ----------------------------------------------------------------------
# SETTINGS
# ----------------------------------------------------------------------
RAW_PATTERNS = ["*messy*.csv", "*study*.csv"]     # messy input file (auto-detected)
CLEAN_FILE = "clean_study_data.csv"               # cleaned output file
DEFAULT_MAX_MARKS = 50        # 98% of rows have max = 50; the rest (100/fifty/?/--) are treated as bad entries
EXAM_YEAR = 2026              # exam dates outside this year are treated as wrong
MAX_STUDY_HOURS = 16          # more study hours than this is unrealistic -> blank
MAX_SLEEP_HOURS = 14

COLS = ["student_id", "subject", "marks", "max_marks", "attendance", "study_hours",
        "difficulty", "exam_date", "sleep_hours", "goal_score", "remarks",
        "orig_index", "entry_time"]

NA_TOKENS = {"", "-", "--", "---", "?", "??", "n/a", "na", "null", "none", "nan",
             "tbd", "not announced", "00-00-0000"}

SUBJECT_MAP = {
    "mathematics": "Mathematics", "maths": "Mathematics", "math": "Mathematics",
    "mathmatics": "Mathematics",
    "physics": "Physics", "physcs": "Physics",
    "chemistry": "Chemistry", "chemistery": "Chemistry", "chem": "Chemistry",
    "programming": "Programming", "programing": "Programming", "prog.": "Programming",
    "prog": "Programming",
    "engineering drawing": "Engineering Drawing",
    "english communication": "English Communication",
}
ABSENT_TOKENS = {"ab", "abs", "absent"}
WORD_NUMBERS = {"zero": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6,
                "seven": 7, "eight": 8, "nine": 9, "ten": 10, "fifty": 50, "hundred": 100}
DIFFICULTY_WORDS = {"very easy": 1, "easy": 2, "med": 3, "medium": 3, "hard": 4, "very hard": 5}

DATE_FORMATS = ["%Y-%m-%d", "%Y/%m/%d", "%y-%m-%d", "%d/%m/%Y", "%d.%m.%Y",
                "%m-%d-%Y", "%d-%b-%Y", "%b %d, %Y", "%B %d %Y"]
TS_FORMATS = ["%Y-%m-%dT%H:%M:%S", "%Y-%m-%d %H:%M", "%d/%m/%Y %H:%M", "%b %d %Y"]


# ----------------------------------------------------------------------
# SMALL HELPERS (clean one value at a time)
# ----------------------------------------------------------------------
def is_na(x):
    return x is None or (isinstance(x, float) and np.isnan(x)) or str(x).strip().lower() in NA_TOKENS


def first_number(x):
    """'23.5 marks' -> 23.5 , '45pts' -> 45 , 'seven' -> 7"""
    if is_na(x):
        return np.nan
    s = str(x).strip().lower()
    if s in WORD_NUMBERS:
        return float(WORD_NUMBERS[s])
    m = re.search(r"-?\d+(?:\.\d+)?", s)
    return float(m.group()) if m else np.nan


def parse_hours(x):
    """'9.5 hrs' / '6.7h' / '324 min' / '10h 30m' -> hours (float)"""
    if is_na(x):
        return np.nan
    s = str(x).strip().lower()
    m = re.fullmatch(r"(\d+(?:\.\d+)?)\s*h\w*\s*(\d+)\s*m\w*", s)       # 10h 30m
    if m:
        return float(m.group(1)) + float(m.group(2)) / 60
    m = re.fullmatch(r"(\d+(?:\.\d+)?)\s*min\w*", s)                      # 324 min
    if m:
        return float(m.group(1)) / 60
    return first_number(s)


def parse_student_id(x):
    """'stu0019', 'STU-0019', 'S0019', ' STU0019\\t' -> 'STU0019'"""
    if is_na(x):
        return np.nan
    s = re.sub(r"[\s\-_]", "", str(x).upper())
    m = re.fullmatch(r"(?:STU|S)?(\d{1,4})", s)
    return f"STU{int(m.group(1)):04d}" if m else np.nan


def parse_subject(x):
    if is_na(x):
        return np.nan
    return SUBJECT_MAP.get(str(x).strip().lower(), np.nan)


def parse_difficulty(x):
    """1-5 scale. 'Hard' -> 4, 'Level 3' -> 3, 10 / -1 -> blank"""
    if is_na(x):
        return np.nan
    s = str(x).strip().lower()
    v = DIFFICULTY_WORDS.get(s, first_number(s))
    return v if v in (1, 2, 3, 4, 5) else np.nan


def parse_date(x, fmts=DATE_FORMATS):
    if is_na(x):
        return pd.NaT
    s = str(x).strip()
    for f in fmts:
        try:
            return pd.Timestamp(pd.to_datetime(s, format=f))
        except (ValueError, TypeError):
            continue
    return pd.NaT


def parse_attendance(x):
    v = first_number(x)
    if np.isnan(v):
        return v
    if 0 <= v <= 1:                    # 0.849 -> 84.9%
        v *= 100
    return v if 0 <= v <= 100 else np.nan   # 150, 999, -10 -> blank


def parse_marks(raw, max_marks):
    """returns (marks, absent_flag)"""
    if is_na(raw):
        return np.nan, False
    s = str(raw).strip().lower()
    if s in ABSENT_TOKENS:
        return np.nan, True
    v = first_number(s)
    if np.isnan(v):
        return np.nan, False
    if "%" in s:                        # '74%' -> 74% of max
        v = v * max_marks / 100
    return (v if 0 <= v <= max_marks else np.nan), False   # 999, -5, 51.5 -> blank


# ----------------------------------------------------------------------
# MAIN CLEANING FUNCTION
# ----------------------------------------------------------------------
def load_raw(path):
    """Returns the raw table after removing title / blank / repeated-header / footer rows."""
    try:
        raw = pd.read_csv(path, header=None, dtype=str, skip_blank_lines=False,
                          keep_default_na=False, encoding="utf-8-sig", encoding_errors="replace")
    except TypeError:                   # older pandas (no encoding_errors)
        raw = pd.read_csv(path, header=None, dtype=str, skip_blank_lines=False,
                          keep_default_na=False, encoding="latin-1")
    # find the real header row ("student id")
    hdr = raw.index[raw[0].str.strip().str.lower() == "student id"]
    if len(hdr) == 0:
        sys.exit("Header row ('student id') not found in the file - please check the file.")
    df = raw.iloc[hdr[0] + 1:, :len(COLS)].copy()
    df.columns = COLS
    df = df.apply(lambda c: c.str.strip())
    # remove header rows that appear again in the middle
    df = df[df["student_id"].str.lower() != "student id"]
    return df.reset_index(drop=True)


def clean_data(path, verbose=False):
    log = {}
    df = load_raw(path)
    log["raw data rows (after header)"] = len(df)

    # 1) empty / separator ('---') / footer ('Total records', 'Exported by admin') rows
    non_na = df.drop(columns="student_id").apply(lambda col: col.map(lambda v: not is_na(v))).sum(axis=1)
    junk = non_na < 5
    log["junk rows removed (blank/---/footer)"] = int(junk.sum())
    df = df[~junk].copy()

    # 2) standardise text columns
    df["student_id"] = df["student_id"].map(parse_student_id)
    df["subject"] = df["subject"].map(parse_subject)
    df["orig_index"] = pd.to_numeric(df["orig_index"], errors="coerce")

    # 3) recover missing student_id / subject from orig_index
    #    (in the file, index = student_no*6 + subject_order; we only use it after verifying against the data)
    known = df.dropna(subset=["student_id", "orig_index"])
    sid_from_idx = (df["orig_index"] // 6 + 1).map(lambda n: f"STU{int(n):04d}" if pd.notna(n) else np.nan)
    agree = (known["student_id"] == sid_from_idx[known.index]).mean() if len(known) else 0
    if agree > 0.98:
        order = (known.dropna(subset=["subject"])
                 .groupby(known["orig_index"] % 6)["subject"].agg(lambda s: s.mode().iloc[0]))
        n_sid = df["student_id"].isna().sum()
        df["student_id"] = df["student_id"].fillna(sid_from_idx)
        n_sub = df["subject"].isna().sum()
        df["subject"] = df["subject"].fillna((df["orig_index"] % 6).map(order))
        log["student_id recovered"] = int(n_sid - df["student_id"].isna().sum())
        log["subject recovered"] = int(n_sub - df["subject"].isna().sum())
    unusable = df["student_id"].isna() | df["subject"].isna()
    log["rows removed (student/subject unknown)"] = int(unusable.sum())
    df = df[~unusable].copy()

    # 4) numeric columns
    df["max_marks"] = DEFAULT_MAX_MARKS
    parsed = [parse_marks(m, DEFAULT_MAX_MARKS) for m in df["marks"]]
    df["absent"] = [p[1] for p in parsed]
    bad_marks = [(not is_na(r) and np.isnan(p[0]) and not p[1]) for r, p in zip(df["marks"], parsed)]
    log["invalid marks blanked (999, -5, >50 ...)"] = int(sum(bad_marks))
    df["marks"] = [p[0] for p in parsed]

    df["attendance"] = df["attendance"].map(parse_attendance)
    df["study_hours"] = df["study_hours"].map(parse_hours)
    df.loc[~df["study_hours"].between(0, MAX_STUDY_HOURS), "study_hours"] = np.nan
    df["sleep_hours"] = df["sleep_hours"].map(parse_hours)
    df.loc[~df["sleep_hours"].between(0, MAX_SLEEP_HOURS), "sleep_hours"] = np.nan
    df["difficulty"] = df["difficulty"].map(parse_difficulty)
    df["goal_score"] = df["goal_score"].map(first_number)
    df.loc[~df["goal_score"].between(0, DEFAULT_MAX_MARKS), "goal_score"] = np.nan   # 500, 'high'

    # 5) dates
    d = df["exam_date"].map(parse_date)
    d = d.where(d.dt.year == EXAM_YEAR)                    # 2025-12-07, 2026-13-45, 31/02 -> NaT
    df["exam_date"] = d
    df["entry_time"] = df["entry_time"].map(lambda v: parse_date(v, TS_FORMATS))   # 'yesterday' -> NaT

    # if exam date is missing/wrong, fill it from the subject (every subject's exam is on a single day)
    subj_date = df.dropna(subset=["exam_date"]).groupby("subject")["exam_date"].agg(lambda s: s.mode().iloc[0])
    n_before = df["exam_date"].isna().sum()
    df["exam_date"] = df["exam_date"].fillna(df["subject"].map(subj_date))
    log["exam_date filled from subject"] = int(n_before - df["exam_date"].isna().sum())

    # 6) remarks
    df["remarks"] = df["remarks"].map(lambda v: "" if is_na(v) else str(v).strip())

    # 7) duplicates: keep the row that is not flagged 'duplicate?' and has more data
    df["_dup_flag"] = df["remarks"].str.lower().eq("duplicate?")
    df["_filled"] = df.notna().sum(axis=1)
    df = df.sort_values(["_dup_flag", "_filled"], ascending=[True, False], kind="stable")
    n = len(df)
    df = df.drop_duplicates(subset=["student_id", "subject"], keep="first")
    log["duplicate rows removed"] = n - len(df)
    df = df.drop(columns=["_dup_flag", "_filled"])

    # 8) derived columns
    df["percentage"] = (df["marks"] / df["max_marks"] * 100).round(1)
    df["grade"] = df["percentage"].map(grade_of)
    both = df["marks"].notna() & df["goal_score"].notna()
    df["goal_met"] = (df["marks"] >= df["goal_score"]).where(both).astype("boolean")   # True/False/blank
    df["status"] = np.select([df["absent"], df["marks"].isna()], ["Absent", "Marks missing"], "Appeared")

    df = df.sort_values(["exam_date", "student_id"]).reset_index(drop=True)
    df = df[["student_id", "subject", "exam_date", "marks", "max_marks", "percentage", "grade",
             "status", "goal_score", "goal_met", "attendance", "study_hours", "sleep_hours",
             "difficulty", "remarks", "entry_time", "orig_index", "absent"]]
    log["final clean rows"] = len(df)
    log["unique students"] = df["student_id"].nunique()
    log["blank (NaN) values per column"] = df.isna().sum()[lambda s: s > 0].to_dict()
    if verbose:
        print("\n=== CLEANING REPORT ===")
        for k, v in log.items():
            print(f"  {k}: {v}")
    return df


def grade_of(p):
    if pd.isna(p):
        return "N/A"
    for cut, g in [(90, "A+"), (80, "A"), (70, "B"), (60, "C"), (50, "D"), (40, "E")]:
        if p >= cut:
            return g
    return "F"


# ----------------------------------------------------------------------
# TERMINAL OUTPUT
# ----------------------------------------------------------------------
def fmt(v, suffix="", nd=1):
    return "N/A" if pd.isna(v) else f"{v:.{nd}f}{suffix}"


def fmt_date(d):
    return "N/A" if pd.isna(d) else d.strftime("%d-%b-%Y (%A)")


def add_class_stats(df):
    g = df.groupby("subject")["marks"]
    df = df.copy()
    df["class_avg"] = g.transform("mean")
    df["class_top"] = g.transform("max")
    df["rank"] = g.rank(ascending=False, method="min")
    df["appeared"] = g.transform("count")
    return df


def show_student_day(r):
    line = "=" * 58
    print(f"\n{line}\n STUDENT : {r.student_id}\n DATE    : {fmt_date(r.exam_date)}\n{line}")
    print(f" Subject        : {r.subject}")
    if r.status == "Absent":
        print(" Result         : ABSENT (did not take the exam)")
    elif r.status == "Marks missing":
        print(" Result         : Marks are not available in the data")
    else:
        print(f" Marks          : {fmt(r.marks)} / {r.max_marks:.0f}   ({fmt(r.percentage, '%')})   Grade: {r.grade}")
        print(f" Rank in subject: {int(r['rank'])} / {int(r.appeared)}   (class avg {fmt(r.class_avg)}, topper {fmt(r.class_top)})")
        diff = r.marks - r.class_avg
        print(f" vs Class avg   : {diff:+.1f} marks ({'above' if diff >= 0 else 'below'})")
    if pd.notna(r.goal_score):
        extra = ""
        if pd.notna(r.marks):
            gap = r.marks - r.goal_score
            extra = f"  -> GOAL MET (+{gap:.1f})" if gap >= 0 else f"  -> {abs(gap):.1f} marks short of goal"
        print(f" Goal score     : {r.goal_score:.0f}{extra}")
    print(f" Attendance     : {fmt(r.attendance, '%')}")
    print(f" Study hours    : {fmt(r.study_hours, ' h')}")
    print(f" Sleep hours    : {fmt(r.sleep_hours, ' h')}")
    print(f" Difficulty     : {fmt(r.difficulty, ' / 5', 0)}")
    print(f" Remarks        : {r.remarks or '-'}")
    print(line)


def show_student_all(sdf, sid):
    line = "=" * 78
    print(f"\n{line}\n STUDENT : {sid}  -  performance across all days\n{line}")
    print(f" {'Date':<12}{'Subject':<22}{'Marks':>7}{'%':>7}{'Grade':>7}{'Goal':>6}{'Rank':>10}  Status")
    for r in sdf.itertuples():
        rank = "-" if pd.isna(r.rank) else f"{int(r.rank)}/{int(r.appeared)}"
        goal = "-" if pd.isna(r.goal_score) else f"{r.goal_score:.0f}"
        print(f" {r.exam_date:%d-%m-%Y}  {r.subject:<22}{fmt(r.marks):>7}{fmt(r.percentage):>7}"
              f"{r.grade:>7}{goal:>6}{rank:>10}  {r.status}")
    print("-" * 78)
    print(f" Average %: {fmt(sdf.percentage.mean(), '%')}   Avg attendance: {fmt(sdf.attendance.mean(), '%')}"
          f"   Avg study: {fmt(sdf.study_hours.mean(), ' h')}   Avg sleep: {fmt(sdf.sleep_hours.mean(), ' h')}")
    ok = sdf.dropna(subset=["percentage"])
    if len(ok):
        b, w = ok.loc[ok.percentage.idxmax()], ok.loc[ok.percentage.idxmin()]
        print(f" Best : {b.subject} ({b.percentage:.1f}%)    Weakest: {w.subject} ({w.percentage:.1f}%)")
    print(line)


def show_day_all(ddf, date, top):
    line = "=" * 70
    subj = ", ".join(sorted(ddf.subject.unique()))
    print(f"\n{line}\n DATE : {fmt_date(date)}   Subject: {subj}\n{line}")
    ok = ddf.dropna(subset=["marks"])
    print(f" Total students: {len(ddf)}   Appeared: {len(ok)}   Absent: {(ddf.status == 'Absent').sum()}"
          f"   Marks missing: {(ddf.status == 'Marks missing').sum()}")
    print(f" Class avg: {fmt(ok.marks.mean())}/50 ({fmt(ok.percentage.mean(), '%')})   "
          f"Highest: {fmt(ok.marks.max())}   Lowest: {fmt(ok.marks.min())}")
    print(f" Goal met by: {int(ddf.goal_met.fillna(False).sum())} students")
    cols = ["student_id", "marks", "percentage", "grade", "attendance", "study_hours"]
    print(f"\n TOP {top}")
    print(ok.nlargest(top, "marks")[cols].to_string(index=False, na_rep="N/A"))
    print(f"\n BOTTOM {top}")
    print(ok.nsmallest(top, "marks")[cols].to_string(index=False, na_rep="N/A"))
    print(line)


# ----------------------------------------------------------------------
# USER INPUT PARSING
# ----------------------------------------------------------------------
WEEKDAYS = ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"]
# Hindi weekday names (kept as accepted input aliases)
WEEKDAY_ALIASES = {"somvar": 0, "mangalvar": 1, "budhvar": 2, "guruvar": 3, "veervar": 3,
                   "shukravar": 4, "shanivar": 5, "ravivar": 6}
MONTHS = {**{m.lower(): i for i, m in enumerate(calendar.month_name) if m},
          **{m.lower(): i for i, m in enumerate(calendar.month_abbr) if m}}


def parse_user_date(s):
    """User's date: '2026-12-18', '18-12-2026', '18/12/2026', '18 Dec 2026', '18 Dec'
    (user input is always interpreted as DD-MM-YYYY)"""
    s = s.strip()
    for f in ("%Y-%m-%d", "%Y/%m/%d", "%d-%m-%Y", "%d/%m/%Y", "%d.%m.%Y",
              "%d %b %Y", "%d %B %Y", "%d-%b-%Y", "%d %b", "%d %B"):
        try:
            d = pd.Timestamp(pd.to_datetime(s, format=f))
        except ValueError:
            continue
        return d.replace(year=EXAM_YEAR) if d.year == 1900 else d   # no year given -> use exam year
    return pd.NaT


def parse_weekday(s):
    """'monday' / 'mon' / 'somvar' -> 0..6"""
    k = s.strip().lower()
    if k in WEEKDAY_ALIASES:
        return WEEKDAY_ALIASES[k]
    for i, w in enumerate(WEEKDAYS):
        if len(k) >= 3 and w.startswith(k):
            return i
    sys.exit(f"Could not understand weekday: {s}  (write monday, tue, somvar ...)")


def parse_month(s):
    """'dec', 'december', '12', '2026-12', 'dec 2026' -> (year or None, month)"""
    k = s.strip().lower()
    m = re.fullmatch(r"(\d{4})[-/](\d{1,2})", k)
    if m:
        return int(m.group(1)), int(m.group(2))
    m = re.fullmatch(r"(\d{1,2})", k)
    if m and 1 <= int(m.group(1)) <= 12:
        return None, int(m.group(1))
    m = re.fullmatch(r"([a-z]+)\s*(\d{4})?", k)
    if m and m.group(1) in MONTHS:
        return (int(m.group(2)) if m.group(2) else None), MONTHS[m.group(1)]
    sys.exit(f"Could not understand month: {s}  (dec, december, 12, 2026-12 ...)")


# ----------------------------------------------------------------------
# STUDY / PERFORMANCE REPORTS  (weekday, month, year, range ...)
# ----------------------------------------------------------------------
def apply_filters(df, a, col):
    f = df.copy()
    label = []
    if a.student_id:
        f = f[f.student_id == a.student_id]
        label.append(a.student_id)
    if a.subject:
        sub = parse_subject(a.subject)
        if pd.isna(sub):
            sys.exit(f"Could not understand subject: {a.subject}  ({', '.join(sorted(set(SUBJECT_MAP.values())))})")
        f = f[f.subject == sub]
        label.append(sub)
    if a.weekday:
        wd = parse_weekday(a.weekday)
        f = f[f[col].dt.dayofweek == wd]
        label.append(WEEKDAYS[wd].title())
    if a.month:
        y, m = parse_month(a.month)
        f = f[f[col].dt.month == m]
        if y:
            f = f[f[col].dt.year == y]
        label.append(f"{calendar.month_name[m]}" + (f" {y}" if y else ""))
    if a.year:
        f = f[f[col].dt.year == a.year]
        label.append(str(a.year))
    lo = parse_user_date(a.date_from) if a.date_from else None
    hi = parse_user_date(a.date_to) if a.date_to else None
    for name, v in (("--from", a.date_from if lo is None else None), ("--to", a.date_to if hi is None else None)):
        if v:
            sys.exit(f"Could not understand {name} date: {v}")
    if lo is not None:
        f = f[f[col] >= lo]
        label.append(f"from {lo:%d-%b-%Y}")
    if hi is not None:
        f = f[f[col] <= hi]
        label.append(f"to {hi:%d-%b-%Y}")
    return f, " | ".join(label) or "ALL DATA"


def group_key(f, col, by):
    if by == "weekday":
        return pd.Categorical(f[col].dt.day_name(), categories=[w.title() for w in WEEKDAYS], ordered=True)
    if by == "month":
        return f[col].dt.strftime("%Y-%m (%B)")
    if by == "year":
        return f[col].dt.year
    if by == "date":
        return f[col].dt.strftime("%Y-%m-%d %a")
    return f["subject"]


def group_table(f, col, by):
    t = (f.groupby(group_key(f, col, by), observed=True)
          .agg(Records=("student_id", "size"), Students=("student_id", "nunique"),
               Study_Total_h=("study_hours", "sum"), Study_Avg_h=("study_hours", "mean"),
               Avg_Marks=("marks", "mean"), Avg_Pct=("percentage", "mean"),
               Attendance=("attendance", "mean"), Sleep_h=("sleep_hours", "mean"))
          .round(1))
    t.index.name = by.title()
    return t


def show_report(f, label, col, by, on):
    line = "=" * 78
    src = "exam date" if on == "exam" else "entry date"
    print(f"\n{line}\n REPORT : {label}      (based on {src})\n{line}")
    if f.empty:
        print(" No records found for this filter.")
        return
    d = f[col]
    print(f" Dates covered   : {d.min():%d-%b-%Y} to {d.max():%d-%b-%Y}  ({d.nunique()} distinct days)")
    print(f" Records         : {len(f)}   Students: {f.student_id.nunique()}   Subjects: {f.subject.nunique()}")
    st = f["study_hours"]
    per_student = f.groupby("student_id")["study_hours"].sum(min_count=1)
    print(f" TOTAL STUDY     : {st.sum():,.1f} hours")
    print(f" Avg study       : {fmt(st.mean(), ' h')} per record   |   {fmt(per_student.mean(), ' h')} per student")
    print(f" Max / Min study : {fmt(st.max(), ' h')} / {fmt(st.min(), ' h')}   (study data missing: {int(st.isna().sum())} records)")
    print(f" Avg marks       : {fmt(f.marks.mean())}/50 ({fmt(f.percentage.mean(), '%')})   "
          f"Attendance: {fmt(f.attendance.mean(), '%')}   Sleep: {fmt(f.sleep_hours.mean(), ' h')}")
    valid = f.dropna(subset=["study_hours", "marks"])
    if len(valid) > 10:
        print(f" Study vs marks  : correlation {valid.study_hours.corr(valid.marks):+.2f}")
    by = by or ("date" if d.nunique() > 1 else None)
    if by:
        print(f"\n BREAKDOWN by {by}\n")
        print(group_table(f, col, by).to_string())
    print(line)


def show_coverage(df, col, on):
    d = df[col].dropna()
    print(f"\n The data ({on} date) contains:")
    print(f"   Weekdays : {', '.join(sorted(d.dt.day_name().unique(), key=lambda w: [x.title() for x in WEEKDAYS].index(w)))}")
    print(f"   Months   : {', '.join(sorted(d.dt.strftime('%B %Y').unique()))}")
    print(f"   Years    : {', '.join(map(str, sorted(d.dt.year.unique())))}")
    print(f"   Range    : {d.min():%d-%b-%Y} to {d.max():%d-%b-%Y}")


# ----------------------------------------------------------------------
# MAIN
# ----------------------------------------------------------------------
def find_input(arg):
    if arg:
        for p in (Path(arg), SCRIPT_DIR / arg):
            if p.exists():
                return p
        sys.exit(f"File not found: {arg}")
    seen = []
    for folder in (Path.cwd(), SCRIPT_DIR):
        for pat in RAW_PATTERNS:
            for p in folder.glob(pat):
                if not p.name.lower().startswith("clean_study_data") and p not in seen:
                    seen.append(p)
    if not seen:
        csvs = sorted({p.name for f in (Path.cwd(), SCRIPT_DIR) for p in f.glob("*.csv")})
        sys.exit("Messy CSV not found.\n"
                 f"  Script folder  : {SCRIPT_DIR}\n  Terminal folder: {Path.cwd()}\n"
                 f"  CSVs in this folder: {csvs or 'none'}\n"
                 "  Put the messy CSV next to the script, or run:  python smart_study.py --file \"full_path.csv\"")
    return max(seen, key=lambda p: p.stat().st_mtime)


def save_clean(df, path):
    out = path.resolve().with_name(CLEAN_FILE)
    data = df.drop(columns=["absent"])
    try:
        data.to_csv(out, index=False, date_format="%Y-%m-%d")
    except PermissionError:             # if it is open in Excel
        out = out.with_name(f"clean_study_data_{datetime.now():%H%M%S}.csv")
        data.to_csv(out, index=False, date_format="%Y-%m-%d")
        print(" Note: clean_study_data.csv is open in Excel/another program, so it was saved under a new name.")
    return out


def build_parser():
    ap = argparse.ArgumentParser(prog="smart_study.py", description="Smart Study Plan - clean + reports",
                                 formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__)
    ap.add_argument("student", nargs="?", help="Student ID (STU0104 or just 104)")
    ap.add_argument("date_pos", nargs="?", help="Date (2026-12-18 / 18-12-2026)")
    ap.add_argument("-d", "--date", help="a single day (all students on that day)")
    ap.add_argument("--weekday", help="monday / tue / somvar ...")
    ap.add_argument("--month", help="dec / december / 12 / 2026-12")
    ap.add_argument("--year", type=int, help="2026")
    ap.add_argument("--from", dest="date_from", help="start date")
    ap.add_argument("--to", dest="date_to", help="end date")
    ap.add_argument("--subject", help="physics / maths / chem ...")
    ap.add_argument("--by", choices=["weekday", "month", "year", "date", "subject"], help="grouped table")
    ap.add_argument("--on", choices=["exam", "entry"], default="exam", help="which date to use (default: exam)")
    ap.add_argument("--dates", action="store_true", help="show available dates")
    ap.add_argument("--clean", action="store_true", help="only run cleaning (no report)")
    ap.add_argument("--top", type=int, default=5, help="how many top/bottom students in the date summary")
    ap.add_argument("--file", help="path to the messy csv (default: auto-detect)")
    return ap


def has_query(a):
    return any([a.student, a.date_pos, a.date, a.weekday, a.month, a.year, a.date_from, a.date_to,
                a.subject, a.by, a.dates, a.on == "entry"])


def run_query(df, args):
    """Runs a single command. On error it calls sys.exit(message)."""
    col = "exam_date" if args.on == "exam" else "entry_time"
    sid_in, date_in = args.student, args.date_pos or args.date
    args.student_id = None
    if sid_in:
        args.student_id = parse_student_id(sid_in)
        if pd.isna(args.student_id):
            sys.exit(f"Could not understand student ID: {sid_in}")
        if args.student_id not in set(df.student_id):
            sys.exit(f"{args.student_id} not found in the data. (Valid: {df.student_id.min()} to {df.student_id.max()})")

    if args.dates:
        print("\nAvailable exam dates:")
        for d, g in df.groupby("exam_date"):
            print(f"  {d:%Y-%m-%d}  {d.strftime('%A'):<10}  {g.subject.iloc[0]}  ({len(g)} students)")
        return

    report_mode = any([args.weekday, args.month, args.year, args.date_from, args.date_to, args.by,
                       args.subject]) or args.on == "entry"
    if report_mode:
        if date_in:                                   # single-day filter
            d = parse_user_date(date_in)
            if pd.isna(d):
                sys.exit(f"Could not understand date: {date_in}")
            args.date_from = args.date_to = d.strftime("%Y-%m-%d")
        f, label = apply_filters(df, args, col)
        missing = int(f[col].isna().sum())
        f = f.dropna(subset=[col])
        show_report(f, label, col, args.by, args.on)
        if missing:
            print(f" Note: {missing} records were skipped because their {args.on} date is not available (e.g. 'yesterday').")
        if f.empty:
            show_coverage(df, col, args.on)
        return

    sid, date = args.student_id, None
    if date_in:
        date = parse_user_date(date_in)
        if pd.isna(date):
            sys.exit(f"Could not understand date: {date_in}  (format: 2026-12-18 or 18-12-2026)")

    if sid and date is not None:
        row = df[(df.student_id == sid) & (df.exam_date == date)]
        if row.empty:
            days = ", ".join(f"{d:%Y-%m-%d}" for d in sorted(df[df.student_id == sid].exam_date.unique()))
            sys.exit(f"{sid} has no exam record on {date:%d-%b-%Y}.\nDates for this student: {days}")
        show_student_day(row.iloc[0])
    elif sid:
        show_student_all(df[df.student_id == sid].sort_values("exam_date"), sid)
    elif date is not None:
        ddf = df[df.exam_date == date]
        if ddf.empty:
            sys.exit(f"No exam on {date:%d-%b-%Y}. Use `--dates` to see the available dates.")
        show_day_all(ddf, date, args.top)


INTERACTIVE_HELP = """
 Now type a command to view student performance (q = quit):
   STU0104 2026-12-18          the student's performance on that day
   STU0104                     the student's performance across all days
   --date 2026-12-11           all students on that day
   --weekday monday            how much study happened on Mondays
   --month dec --by weekday    the whole month
   --year 2026                 the whole year
   --dates                     available exam dates
"""


def main():
    ap = build_parser()
    args = ap.parse_args()
    path = find_input(args.file)
    interactive = not has_query(args) and not args.clean

    try:
        df = clean_data(path, verbose=args.clean or interactive)
    except Exception as e:                           # show a clean error, not a traceback
        sys.exit(f"Error during cleaning: {type(e).__name__}: {e}\n"
                 f"File: {path}\nPandas version: {pd.__version__}  (try: pip install -U pandas)")
    out = save_clean(df, path)
    print(f"\n >>> CLEAN FILE SAVED: {out}   ({len(df)} rows)")

    if args.clean:
        return
    df = add_class_stats(df)

    if not interactive:
        run_query(df, args)
        return

    print(INTERACTIVE_HELP)
    while True:
        try:
            line = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if line.lower() in ("q", "quit", "exit"):
            break
        if not line:
            continue
        try:
            run_query(df, ap.parse_args(shlex.split(line)))
        except SystemExit as e:                      # on bad input, don't quit - just show the message
            if e.code not in (None, 0, 2):
                print(f" {e.code}")

if __name__ == "__main__":
    import sys
    # If run with arguments like: python dataclean.py STU0105
    if len(sys.argv) > 1:
        main()
    else:
        # Interactive mode - just enter ID
        print("=== Smart Study Plan - Student Progress ===")
        print("Type 'exit' to quit\n")
        while True:
            sid = input("Enter Student ID (e.g. STU0105): ").strip()
            if sid.lower() == 'exit' or sid == '':
                print("Exiting...")
                break
            
            date_input = input("Enter Date [Press Enter for 2026-12-18]: ").strip()
            if not date_input:
                date_input = "2026-12-18"
            
            sys.argv = ["dataclean.py", sid, date_input]
            try:
                main()
            except SystemExit:
                pass
            except Exception as e:
                print(f"Error: {e}")