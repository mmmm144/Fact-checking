"""Shared utilities for the eight EDA notebooks in this directory.

The functions intentionally keep data loading, cleaning and plotting in one
place so every publisher is analysed with the same definitions.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from urllib.parse import urlparse

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from IPython.display import Markdown, display


EXPECTED_COLUMNS = [
    "id",
    "source_type",
    "source_name",
    "url",
    "domain",
    "publish_date",
    "claim",
    "original_text",
    "label",
    "evidence",
    "justification",
    "comments",
]

SOURCE_CONFIGS = {
    "gso": {
        "title": "GSO/NSO Việt Nam",
        "path": "loc_crawl/output/gso.json",
    },
    "moh": {
        "title": "Bộ Y tế (MOH)",
        "path": "loc_crawl/output/moh.json",
    },
    "vafc": {
        "title": "VAFC / tingia.gov.vn",
        "path": "loc_crawl/output/vafc.json",
    },
    "who": {
        "title": "World Health Organization (WHO)",
        "path": "loc_crawl/output/who.json",
    },
    "bao_chinh_phu": {
        "title": "Báo Chính phủ",
        "path": "loc_crawl/output_news/bao_chinh_phu.json",
    },
    "vnexpress": {
        "title": "VnExpress",
        "path": "loc_crawl/output_news/vnexpress.json",
    },
    "world_bank": {
        "title": "World Bank",
        "path": "loc_crawl/output_news/world_bank.json",
    },
}

SOURCE_ORDER = list(SOURCE_CONFIGS)

PALETTE = {
    key: color
    for key, color in zip(
        SOURCE_ORDER,
        sns.color_palette("colorblind", n_colors=len(SOURCE_ORDER)).as_hex(),
    )
}


def setup_notebook() -> None:
    """Set compact plotting and pandas display defaults."""
    sns.set_theme(style="whitegrid", context="notebook")
    plt.rcParams.update(
        {
            "figure.dpi": 110,
            "axes.titleweight": "bold",
            "axes.titlesize": 12,
            "axes.labelsize": 10,
            "figure.autolayout": True,
        }
    )
    pd.set_option("display.max_columns", 30)
    pd.set_option("display.max_colwidth", 100)
    pd.set_option("display.float_format", lambda value: f"{value:,.2f}")


def find_project_root(start: str | Path | None = None) -> Path:
    """Find the repository root whether Jupyter starts at root or notebook dir."""
    start_path = Path(start or Path.cwd()).resolve()
    for candidate in (start_path, *start_path.parents):
        if (candidate / "loc_crawl").is_dir():
            return candidate
    raise FileNotFoundError(
        "Không tìm thấy thư mục 'loc_crawl'. "
        "Hãy chạy notebook từ repository Fact-checking hoặc thư mục con của nó."
    )


def _clean_scalar(value) -> str:
    if value is None:
        return ""
    if isinstance(value, float) and np.isnan(value):
        return ""
    return str(value).strip()


def _word_count(value) -> int:
    text = _clean_scalar(value)
    return len(re.findall(r"\S+", text)) if text else 0


def _comment_count(value) -> int:
    if isinstance(value, (list, tuple, dict)):
        return len(value)
    return 0 if not _clean_scalar(value) else 1


def _split_domains(value) -> list[str]:
    text = _clean_scalar(value)
    if not text:
        return ["(Thiếu domain)"]
    domains = [part.strip() for part in text.split(";") if part.strip()]
    return domains or ["(Thiếu domain)"]


def _hostname(value) -> str:
    text = _clean_scalar(value)
    try:
        return urlparse(text).netloc.lower().removeprefix("www.")
    except (TypeError, ValueError):
        return ""


def _valid_http_url(value) -> bool:
    text = _clean_scalar(value)
    try:
        parsed = urlparse(text)
        return parsed.scheme in {"http", "https"} and bool(parsed.netloc)
    except (TypeError, ValueError):
        return False


def _duplicate_nonempty_count(values: pd.Series) -> int:
    """Count repeated rows while excluding blank values from duplicate checks."""
    nonempty = values.map(_clean_scalar)
    nonempty = nonempty[nonempty.ne("")]
    return int(nonempty.duplicated().sum())

def prepare_dataframe(raw: pd.DataFrame, source_key: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Normalize derived fields and return article-level and atomic-domain tables."""
    df = raw.copy()
    for column in EXPECTED_COLUMNS:
        if column not in df.columns:
            df[column] = np.nan

    text_columns = [column for column in EXPECTED_COLUMNS if column != "comments"]
    for column in text_columns:
        df[column] = df[column].map(_clean_scalar)

    df.insert(0, "source_key", source_key)
    df.insert(1, "publisher", SOURCE_CONFIGS[source_key]["title"])
    df["_row_number"] = np.arange(1, len(df) + 1)
    df["record_key"] = source_key + "::" + df["_row_number"].astype(str)

    date_text = df["publish_date"].replace("", pd.NA)
    df["date"] = pd.to_datetime(date_text, errors="coerce")
    df["year"] = df["date"].dt.year.astype("Int64")
    df["month"] = df["date"].dt.to_period("M").astype("string")
    df["url_host"] = df["url"].map(_hostname)
    df["valid_http_url"] = df["url"].map(_valid_http_url)
    df["domain_count"] = df["domain"].map(lambda value: len(_split_domains(value)))
    df["comments_count"] = df["comments"].map(_comment_count)

    for column in ["claim", "original_text", "evidence", "justification"]:
        df[f"{column}_chars"] = df[column].str.len()
        df[f"{column}_words"] = df[column].map(_word_count)

    domains = (
        df[["record_key", "source_key", "publisher", "year", "date", "domain"]]
        .assign(domain_atomic=lambda frame: frame["domain"].map(_split_domains))
        .explode("domain_atomic", ignore_index=True)
    )
    domains["domain_atomic"] = domains["domain_atomic"].astype("string")
    return df, domains


def load_source(
    source_key: str, project_root: str | Path | None = None
) -> tuple[pd.DataFrame, pd.DataFrame, Path]:
    """Load one configured JSON source."""
    if source_key not in SOURCE_CONFIGS:
        raise KeyError(f"Nguồn không hợp lệ: {source_key!r}")
    root = find_project_root(project_root)
    path = root / SOURCE_CONFIGS[source_key]["path"]
    if not path.exists():
        raise FileNotFoundError(f"Không tìm thấy dữ liệu: {path}")
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, list):
        raise TypeError(f"JSON phải có root là list, nhận được {type(payload).__name__}")
    raw = pd.DataFrame(payload)
    df, domains = prepare_dataframe(raw, source_key)
    return df, domains, path


def load_all_sources(
    project_root: str | Path | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load and combine all configured sources."""
    article_frames = []
    domain_frames = []
    file_rows = []
    for source_key in SOURCE_ORDER:
        df, domains, path = load_source(source_key, project_root)
        article_frames.append(df)
        domain_frames.append(domains)
        file_rows.append(
            {
                "source_key": source_key,
                "publisher": SOURCE_CONFIGS[source_key]["title"],
                "file": str(path),
                "file_mb": path.stat().st_size / (1024**2),
            }
        )
    articles = pd.concat(article_frames, ignore_index=True, sort=False)
    atomic_domains = pd.concat(domain_frames, ignore_index=True, sort=False)
    files = pd.DataFrame(file_rows)
    return articles, atomic_domains, files


def _missing_table(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for column in EXPECTED_COLUMNS:
        if column == "comments":
            missing = df[column].map(
                lambda value: value is None
                or (isinstance(value, float) and np.isnan(value))
                or (isinstance(value, (list, tuple, dict)) and len(value) == 0)
                or (
                    not isinstance(value, (list, tuple, dict))
                    and _clean_scalar(value) == ""
                )
            )
        else:
            missing = df[column].map(_clean_scalar).eq("")
        rows.append(
            {
                "field": column,
                "missing": int(missing.sum()),
                "missing_pct": float(missing.mean() * 100),
            }
        )
    return pd.DataFrame(rows).sort_values(
        ["missing", "field"], ascending=[False, True], ignore_index=True
    )


def source_overview(
    df: pd.DataFrame, domains: pd.DataFrame, path: Path | None = None
) -> pd.DataFrame:
    """Display headline metrics and a small raw-data preview."""
    valid_dates = df["date"].dropna()
    metrics = pd.DataFrame(
        {
            "Chỉ số": [
                "Số bài",
                "Số domain nguyên bản",
                "Số domain sau khi tách",
                "Bài đa-domain",
                "Ngày hợp lệ",
                "Ngày nhỏ nhất",
                "Ngày lớn nhất",
                "ID trùng",
                "URL trùng",
            ],
            "Giá trị": [
                f"{len(df):,}",
                f"{df['domain'].replace('', pd.NA).nunique(dropna=True):,}",
                f"{domains['domain_atomic'].replace('(Thiếu domain)', pd.NA).nunique(dropna=True):,}",
                f"{df['domain_count'].gt(1).sum():,}",
                f"{df['date'].notna().sum():,} ({df['date'].notna().mean():.1%})",
                valid_dates.min().date().isoformat() if len(valid_dates) else "—",
                valid_dates.max().date().isoformat() if len(valid_dates) else "—",
                f"{_duplicate_nonempty_count(df['id']):,}",
                f"{_duplicate_nonempty_count(df['url']):,}",
            ],
        }
    )
    if path is not None:
        display(Markdown(f"**File:** `{path}`  \n**Kích thước:** {path.stat().st_size / (1024**2):,.2f} MB"))
    display(metrics)
    display(
        df[
            [
                "id",
                "source_name",
                "domain",
                "publish_date",
                "claim",
                "label",
                "url",
            ]
        ].head(5)
    )
    return metrics


def source_domain_analysis(
    df: pd.DataFrame, domains: pd.DataFrame, top_n: int = 25
) -> pd.DataFrame:
    """Show raw and atomic domain distributions plus multi-domain diagnostics."""
    raw_counts = (
        df["domain"]
        .replace("", "(Thiếu domain)")
        .value_counts(dropna=False)
        .rename_axis("domain_nguyên_bản")
        .reset_index(name="số_bài")
    )
    atomic_counts = (
        domains["domain_atomic"]
        .value_counts(dropna=False)
        .rename_axis("domain")
        .reset_index(name="số_bài")
    )
    atomic_counts["tỷ_lệ_trên_bài"] = atomic_counts["số_bài"] / len(df)
    display(Markdown("**Phân bố domain sau khi tách dấu `;`**"))
    display(atomic_counts.head(top_n))

    shown = atomic_counts.head(top_n).sort_values("số_bài")
    fig, ax = plt.subplots(figsize=(10, max(4, 0.42 * len(shown))))
    sns.barplot(data=shown, x="số_bài", y="domain", color="#3b82f6", ax=ax)
    ax.set(title=f"Top {min(top_n, len(shown))} domain", xlabel="Số bài", ylabel="")
    for container in ax.containers:
        ax.bar_label(container, fmt="%d", padding=3, fontsize=8)
    plt.show()

    if df["domain_count"].gt(1).any():
        display(Markdown("**Tổ hợp domain nguyên bản phổ biến**"))
        display(raw_counts.head(15))
        multi = (
            df["domain_count"]
            .value_counts()
            .sort_index()
            .rename_axis("số_domain_trên_bài")
            .reset_index(name="số_bài")
        )
        display(multi)
    return atomic_counts


def source_timeline(df: pd.DataFrame, domains: pd.DataFrame) -> None:
    """Analyse date coverage, yearly volume and domain composition over time."""
    valid = df.dropna(subset=["date"]).copy()
    invalid_count = len(df) - len(valid)
    display(
        pd.DataFrame(
            {
                "trạng_thái_ngày": ["Hợp lệ", "Thiếu/không parse được"],
                "số_bài": [len(valid), invalid_count],
                "tỷ_lệ": [len(valid) / len(df), invalid_count / len(df)],
            }
        )
    )
    if valid.empty:
        display(Markdown("> Không có `publish_date` hợp lệ nên bỏ qua biểu đồ thời gian."))
        return

    yearly = (
        valid.groupby("year", dropna=False)
        .size()
        .rename("số_bài")
        .reset_index()
        .sort_values("year")
    )
    display(yearly)
    fig, axes = plt.subplots(1, 2, figsize=(14, 4.5))
    sns.barplot(data=yearly, x="year", y="số_bài", color="#0ea5e9", ax=axes[0])
    axes[0].set(title="Số bài theo năm", xlabel="Năm", ylabel="Số bài")
    axes[0].tick_params(axis="x", rotation=45)

    monthly = (
        valid.set_index("date")
        .resample("MS")
        .size()
        .rename("số_bài")
        .reset_index()
    )
    sns.lineplot(data=monthly, x="date", y="số_bài", marker="o", ax=axes[1])
    axes[1].set(title="Số bài theo tháng", xlabel="Tháng", ylabel="Số bài")
    plt.show()

    domain_year = (
        domains.dropna(subset=["year"])
        .groupby(["domain_atomic", "year"])
        .size()
        .unstack(fill_value=0)
    )
    if not domain_year.empty:
        top_domains = domains["domain_atomic"].value_counts().head(15).index
        domain_year = domain_year.reindex(top_domains).fillna(0)
        fig, ax = plt.subplots(
            figsize=(max(8, 0.75 * len(domain_year.columns)), max(4, 0.42 * len(domain_year)))
        )
        sns.heatmap(domain_year, cmap="Blues", annot=True, fmt=".0f", ax=ax)
        ax.set(title="Domain theo năm (top 15)", xlabel="Năm", ylabel="")
        plt.show()


def source_quality_analysis(df: pd.DataFrame) -> pd.DataFrame:
    """Display missingness, duplicates, URL validity and possible anomalies."""
    missing = _missing_table(df)
    display(missing)

    duplicate_checks = pd.DataFrame(
        {
            "khóa": ["id", "url", "claim", "original_text"],
            "số_dòng_trùng": [
                _duplicate_nonempty_count(df[column])
                for column in ["id", "url", "claim", "original_text"]
            ],
            "số_giá_trị_duy_nhất": [
                int(df[column].replace("", pd.NA).nunique(dropna=True))
                for column in ["id", "url", "claim", "original_text"]
            ],
        }
    )
    display(Markdown("**Trùng lặp và URL**"))
    display(duplicate_checks)
    display(
        pd.DataFrame(
            {
                "chỉ_số": [
                    "URL HTTP(S) hợp lệ",
                    "URL không hợp lệ/thiếu",
                    "Ngày ở tương lai so với hôm nay",
                    "Bài có comments",
                ],
                "số_bài": [
                    int(df["valid_http_url"].sum()),
                    int((~df["valid_http_url"]).sum()),
                    int(df["date"].gt(pd.Timestamp.today().normalize()).sum()),
                    int(df["comments_count"].gt(0).sum()),
                ],
            }
        )
    )

    fig, ax = plt.subplots(figsize=(10, 4))
    plot_data = missing.sort_values("missing_pct", ascending=False)
    sns.barplot(data=plot_data, x="field", y="missing_pct", color="#f97316", ax=ax)
    ax.set(title="Tỷ lệ thiếu theo trường", xlabel="", ylabel="Thiếu (%)")
    ax.tick_params(axis="x", rotation=45)
    plt.show()
    return missing


def source_text_analysis(df: pd.DataFrame, domains: pd.DataFrame) -> pd.DataFrame:
    """Summarise text lengths and compare original-text length by domain."""
    length_columns = [
        "claim_words",
        "original_text_words",
        "evidence_words",
        "justification_words",
    ]
    summary = (
        df[length_columns]
        .describe(percentiles=[0.25, 0.5, 0.75, 0.9, 0.95, 0.99])
        .T.rename_axis("trường")
    )
    display(summary)

    long_lengths = df[length_columns].melt(var_name="trường", value_name="số_từ")
    upper = max(1, float(long_lengths["số_từ"].quantile(0.99)))
    fig, axes = plt.subplots(1, 2, figsize=(14, 4.5))
    sns.boxplot(data=long_lengths, x="trường", y="số_từ", showfliers=False, ax=axes[0])
    axes[0].set(title="Độ dài văn bản (không vẽ ngoại lệ)", xlabel="", ylabel="Số từ")
    axes[0].tick_params(axis="x", rotation=25)
    sns.histplot(
        data=df,
        x="original_text_words",
        bins=40,
        binrange=(0, upper),
        color="#8b5cf6",
        ax=axes[1],
    )
    axes[1].set(
        title="Phân bố độ dài original_text (đến percentile 99)",
        xlabel="Số từ",
        ylabel="Số bài",
    )
    plt.show()

    top_domains = domains["domain_atomic"].value_counts().head(15).index
    domain_lengths = domains[["record_key", "domain_atomic"]].merge(
        df[["record_key", "original_text_words"]], on="record_key", how="left"
    )
    domain_summary = (
        domain_lengths[domain_lengths["domain_atomic"].isin(top_domains)]
        .groupby("domain_atomic")["original_text_words"]
        .agg(số_bài="size", trung_vị="median", trung_bình="mean", p90=lambda x: x.quantile(0.9))
        .sort_values("trung_vị", ascending=False)
    )
    display(Markdown("**Độ dài `original_text` theo top domain**"))
    display(domain_summary)
    return summary


def source_metadata_analysis(df: pd.DataFrame) -> None:
    """Show label/source/host distributions."""
    for column, title in [
        ("label", "Nhãn"),
        ("source_type", "Loại nguồn"),
        ("source_name", "Tên nguồn"),
        ("url_host", "Hostname"),
    ]:
        table = (
            df[column]
            .replace("", "(Thiếu)")
            .value_counts(dropna=False)
            .rename_axis(column)
            .reset_index(name="số_bài")
        )
        table["tỷ_lệ"] = table["số_bài"] / len(df)
        display(Markdown(f"**{title}**"))
        display(table.head(20))


def source_samples(df: pd.DataFrame, n_per_domain: int = 2) -> pd.DataFrame:
    """Show deterministic samples from each raw domain."""
    samples = (
        df.sort_values(["domain", "date", "id"], na_position="last")
        .groupby("domain", dropna=False, group_keys=False)
        .head(n_per_domain)
    )
    columns = ["domain", "publish_date", "claim", "label", "url"]
    display(samples[columns].reset_index(drop=True))
    return samples[columns].reset_index(drop=True)


def source_findings(df: pd.DataFrame, domains: pd.DataFrame) -> None:
    """Generate concise, data-driven observations."""
    top_domain = domains["domain_atomic"].value_counts().index[0]
    top_count = int(domains["domain_atomic"].value_counts().iloc[0])
    missing_dates = int(df["date"].isna().sum())
    multi = int(df["domain_count"].gt(1).sum())
    labels = ", ".join(
        f"{label or '(thiếu)'}: {count:,}"
        for label, count in df["label"].value_counts(dropna=False).items()
    )
    display(
        Markdown(
            "\n".join(
                [
                    "### Nhận xét tự động",
                    f"- Có **{len(df):,}** bài và **{domains['domain_atomic'].nunique():,}** domain sau khi tách.",
                    f"- Domain lớn nhất là **{top_domain}** với **{top_count:,}** lượt gán.",
                    f"- Có **{multi:,}** bài đa-domain và **{missing_dates:,}** bài thiếu/không parse được ngày.",
                    f"- Phân bố nhãn: **{labels}**.",
                    "- Các nhận xét này là mô tả dữ liệu, không phải kết luận về tính đại diện hay chất lượng nội dung.",
                ]
            )
        )
    )


def combined_overview(
    articles: pd.DataFrame, domains: pd.DataFrame, files: pd.DataFrame
) -> pd.DataFrame:
    """Publisher-level counts and headline data-quality indicators."""
    grouped = articles.groupby(["source_key", "publisher"], sort=False)
    overview = grouped.agg(
        số_bài=("record_key", "size"),
        domain_nguyên_bản=("domain", lambda x: x.replace("", pd.NA).nunique(dropna=True)),
        ngày_hợp_lệ=("date", lambda x: int(x.notna().sum())),
        ngày_nhỏ_nhất=("date", "min"),
        ngày_lớn_nhất=("date", "max"),
        id_trùng=("id", _duplicate_nonempty_count),
        url_trùng=("url", _duplicate_nonempty_count),
        url_hợp_lệ=("valid_http_url", "sum"),
        trung_vị_từ=("original_text_words", "median"),
    ).reset_index()
    atomic_unique = (
        domains.groupby("source_key")["domain_atomic"]
        .nunique()
        .rename("domain_sau_tách")
        .reset_index()
    )
    overview = (
        overview.merge(atomic_unique, on="source_key", how="left")
        .merge(files[["source_key", "file_mb"]], on="source_key", how="left")
    )
    overview["tỷ_lệ_ngày_hợp_lệ"] = overview["ngày_hợp_lệ"] / overview["số_bài"]
    overview["tỷ_lệ_url_hợp_lệ"] = overview["url_hợp_lệ"] / overview["số_bài"]
    overview["ngày_nhỏ_nhất"] = overview["ngày_nhỏ_nhất"].dt.date
    overview["ngày_lớn_nhất"] = overview["ngày_lớn_nhất"].dt.date
    overview = overview[
        [
            "publisher",
            "số_bài",
            "domain_nguyên_bản",
            "domain_sau_tách",
            "tỷ_lệ_ngày_hợp_lệ",
            "ngày_nhỏ_nhất",
            "ngày_lớn_nhất",
            "id_trùng",
            "url_trùng",
            "tỷ_lệ_url_hợp_lệ",
            "trung_vị_từ",
            "file_mb",
        ]
    ]
    display(overview)

    plot_data = overview.sort_values("số_bài")
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.barplot(data=plot_data, x="số_bài", y="publisher", palette="colorblind", hue="publisher", legend=False, ax=ax)
    ax.set(title=f"Số bài theo nguồn — tổng {len(articles):,} bài", xlabel="Số bài", ylabel="")
    for container in ax.containers:
        ax.bar_label(container, fmt="%d", padding=3, fontsize=8)
    plt.show()
    return overview


def combined_domain_analysis(
    articles: pd.DataFrame, domains: pd.DataFrame, top_n: int = 20
) -> pd.DataFrame:
    """Compare domain counts within and across publishers."""
    counts = (
        domains.groupby(["source_key", "publisher", "domain_atomic"])
        .agg(số_bài=("record_key", "nunique"))
        .reset_index()
    )
    source_totals = articles.groupby("source_key")["record_key"].size()
    counts["tỷ_lệ_trong_nguồn"] = counts["số_bài"] / counts["source_key"].map(source_totals)
    counts["xếp_hạng_trong_nguồn"] = counts.groupby("source_key")["số_bài"].rank(
        method="dense", ascending=False
    )
    display(
        counts.sort_values(["source_key", "số_bài"], ascending=[True, False])
        .groupby("source_key", group_keys=False)
        .head(top_n)
        .reset_index(drop=True)
    )

    fig, axes = plt.subplots(
        len(SOURCE_ORDER),
        1,
        figsize=(12, 3.4 * len(SOURCE_ORDER)),
        constrained_layout=True,
    )
    for ax, source_key in zip(axes, SOURCE_ORDER):
        subset = (
            counts[counts["source_key"].eq(source_key)]
            .nlargest(15, "số_bài")
            .sort_values("số_bài")
        )
        sns.barplot(
            data=subset,
            x="số_bài",
            y="domain_atomic",
            color=PALETTE[source_key],
            ax=ax,
        )
        ax.set(
            title=SOURCE_CONFIGS[source_key]["title"],
            xlabel="Số bài",
            ylabel="",
        )
    plt.show()

    top_domains = (
        counts.sort_values(["source_key", "số_bài"], ascending=[True, False])
        .groupby("source_key", group_keys=False)
        .head(10)
    )
    matrix = top_domains.pivot_table(
        index="domain_atomic",
        columns="publisher",
        values="số_bài",
        fill_value=0,
        aggfunc="sum",
    )
    fig, ax = plt.subplots(
        figsize=(13, max(6, 0.35 * len(matrix)))
    )
    sns.heatmap(matrix, cmap="YlGnBu", annot=True, fmt=".0f", linewidths=0.3, ax=ax)
    ax.set(title="Ma trận số bài: top 10 domain của mỗi nguồn", xlabel="", ylabel="")
    plt.show()
    return counts


def combined_quality_analysis(articles: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Compare missing fields and duplicate indicators across publishers."""
    missing_rows = []
    for (source_key, publisher), frame in articles.groupby(
        ["source_key", "publisher"], sort=False
    ):
        table = _missing_table(frame)
        table["source_key"] = source_key
        table["publisher"] = publisher
        missing_rows.append(table)
    missing = pd.concat(missing_rows, ignore_index=True)

    missing_matrix = missing.pivot(
        index="publisher", columns="field", values="missing_pct"
    ).reindex([SOURCE_CONFIGS[key]["title"] for key in SOURCE_ORDER])
    display(missing_matrix)
    fig, ax = plt.subplots(figsize=(14, 5))
    sns.heatmap(
        missing_matrix,
        cmap="OrRd",
        annot=True,
        fmt=".1f",
        vmin=0,
        vmax=100,
        ax=ax,
    )
    ax.set(title="Tỷ lệ thiếu theo nguồn và trường (%)", xlabel="", ylabel="")
    plt.show()

    duplicates = (
        articles.groupby(["source_key", "publisher"], sort=False)
        .agg(
            id_trùng=("id", _duplicate_nonempty_count),
            url_trùng=("url", _duplicate_nonempty_count),
            claim_trùng=("claim", _duplicate_nonempty_count),
            original_text_trùng=(
                "original_text",
                _duplicate_nonempty_count,
            ),
            url_không_hợp_lệ=("valid_http_url", lambda x: int((~x).sum())),
            ngày_tương_lai=(
                "date",
                lambda x: int(x.gt(pd.Timestamp.today().normalize()).sum()),
            ),
        )
        .reset_index()
    )
    display(duplicates)
    return missing, duplicates


def combined_timeline(articles: pd.DataFrame) -> pd.DataFrame:
    """Compare yearly/monthly publication coverage."""
    valid = articles.dropna(subset=["date"]).copy()
    yearly = (
        valid.groupby(["source_key", "publisher", "year"])
        .size()
        .rename("số_bài")
        .reset_index()
    )
    display(yearly)
    if yearly.empty:
        display(Markdown("> Không có ngày hợp lệ ở bất kỳ nguồn nào."))
        return yearly

    fig, ax = plt.subplots(figsize=(14, 6))
    sns.lineplot(
        data=yearly,
        x="year",
        y="số_bài",
        hue="publisher",
        marker="o",
        palette=[PALETTE[key] for key in SOURCE_ORDER],
        ax=ax,
    )
    ax.set(title="Số bài theo năm và nguồn", xlabel="Năm", ylabel="Số bài")
    ax.legend(title="", bbox_to_anchor=(1.02, 1), loc="upper left")
    plt.show()

    recent = valid[valid["date"].ge(valid["date"].max() - pd.DateOffset(years=2))]
    monthly = (
        recent.assign(month_date=recent["date"].dt.to_period("M").dt.to_timestamp())
        .groupby(["source_key", "publisher", "month_date"])
        .size()
        .rename("số_bài")
        .reset_index()
    )
    fig, ax = plt.subplots(figsize=(14, 6))
    sns.lineplot(
        data=monthly,
        x="month_date",
        y="số_bài",
        hue="publisher",
        palette=[PALETTE[key] for key in SOURCE_ORDER],
        ax=ax,
    )
    ax.set(title="Số bài theo tháng trong 2 năm gần nhất có dữ liệu", xlabel="Tháng", ylabel="Số bài")
    ax.legend(title="", bbox_to_anchor=(1.02, 1), loc="upper left")
    plt.show()
    return yearly


def combined_text_analysis(articles: pd.DataFrame) -> pd.DataFrame:
    """Compare text-length distributions across publishers."""
    summary = (
        articles.groupby(["source_key", "publisher"], sort=False)
        .agg(
            số_bài=("record_key", "size"),
            claim_trung_vị=("claim_words", "median"),
            original_trung_vị=("original_text_words", "median"),
            original_trung_bình=("original_text_words", "mean"),
            original_p90=("original_text_words", lambda x: x.quantile(0.9)),
            justification_trung_vị=("justification_words", "median"),
        )
        .reset_index()
    )
    display(summary)

    upper = max(1, float(articles["original_text_words"].quantile(0.99)))
    fig, ax = plt.subplots(figsize=(14, 6))
    sns.boxplot(
        data=articles,
        x="publisher",
        y="original_text_words",
        hue="publisher",
        palette=[PALETTE[key] for key in SOURCE_ORDER],
        legend=False,
        showfliers=False,
        ax=ax,
    )
    ax.set_ylim(0, upper)
    ax.set(
        title="Độ dài original_text theo nguồn (giới hạn ở percentile 99)",
        xlabel="",
        ylabel="Số từ",
    )
    ax.tick_params(axis="x", rotation=30)
    plt.show()
    return summary


def combined_metadata_analysis(articles: pd.DataFrame) -> None:
    """Compare label, source type and hostname distributions."""
    for column, title in [
        ("label", "Nhãn theo nguồn"),
        ("source_type", "Loại nguồn"),
        ("url_host", "Hostname"),
    ]:
        table = (
            articles.groupby(["publisher", column], dropna=False)
            .size()
            .rename("số_bài")
            .reset_index()
            .sort_values(["publisher", "số_bài"], ascending=[True, False])
        )
        display(Markdown(f"**{title}**"))
        display(table)


def combined_findings(articles: pd.DataFrame, domains: pd.DataFrame) -> None:
    """Generate concise cross-source observations."""
    source_counts = articles["publisher"].value_counts()
    date_coverage = (
        articles.groupby("publisher")["date"]
        .apply(lambda x: x.notna().mean())
        .sort_values()
    )
    multi_counts = articles.groupby("publisher")["domain_count"].apply(lambda x: x.gt(1).sum())
    largest_source = source_counts.index[0]
    lowest_date_source = date_coverage.index[-1]
    display(
        Markdown(
            "\n".join(
                [
                    "### Nhận xét tự động",
                    f"- Tổng cộng **{len(articles):,}** bài từ **{articles['publisher'].nunique()}** nguồn.",
                    f"- Nguồn có nhiều bài nhất: **{largest_source}** ({source_counts.iloc[0]:,} bài).",
                    f"- Nguồn có độ phủ ngày thấp nhất: **{lowest_date_source}** ({date_coverage.iloc[-1]:.1%}).",
                    f"- Có **{int(multi_counts.sum()):,}** bài đa-domain; cần dùng bảng domain đã tách khi đếm chủ đề.",
                    f"- Toàn bộ dữ liệu có **{domains['domain_atomic'].nunique():,}** tên domain khác nhau sau khi tách.",
                    "- So sánh số lượng giữa các nguồn phản ánh phạm vi crawler hiện tại, không mặc nhiên phản ánh quy mô xuất bản thực tế.",
                ]
            )
        )
    )


setup_notebook()
