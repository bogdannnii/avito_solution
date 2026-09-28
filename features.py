#Построение признаков куки по событиям внутри суточного окна наблюдения
# Все признаки считаются только по событиям с window_start_ts <= event_ts < window_end_ts, т.е. доступны на момент окончания окна
# События после окна не используются - это утечка будущего
from __future__ import annotations

import numpy as np
import pandas as pd

DATA_DIR = "data"
TS_COLS = ["cookie_created_at", "window_start_ts", "window_end_ts"]

EVENT_TYPES = [
    "search_results_view", "item_view", "photo_swipe", "seller_page_view",
    "contact_phone_show", "contact_chat_open", "contact_message_sent",
    "favorite_add", "login",
]
CONTACT_TYPES = ["contact_phone_show", "contact_chat_open", "contact_message_sent"]
SESSION_GAP_S = 30 * 60
POP_LOOKBACK_DAYS = 7


def load_data(data_dir: str = DATA_DIR):
    train = pd.read_csv(f"{data_dir}/train.csv", parse_dates=TS_COLS)
    test = pd.read_csv(f"{data_dir}/test.csv", parse_dates=TS_COLS)
    events = pd.read_csv(f"{data_dir}/events.csv.gz", parse_dates=["event_ts"])
    return train, test, events


# 'WEB'/'Web'/'web'/'desktop', 'ANDROID'/'Android'/..., 'IOS'/'iOS'/'iphone' — одно и то же
def normalize_platform(p: pd.Series) -> pd.Series:
    p = p.str.strip().str.lower()
    return p.replace({"desktop": "web", "iphone": "ios"})


#Сырую строку UA не берём как категорию, вытаскиваем из неё семейство клиента и мажорную версию.
def parse_user_agent(ua: pd.Series) -> pd.DataFrame:
    family = np.select(
        [
            ua.str.contains("HeadlessChrome", regex=False),
            ua.str.startswith("Avito/"),
            ua.str.contains("YaBrowser", regex=False),
            ua.str.contains("Firefox", regex=False),
            ua.str.contains("iPhone", regex=False),
            ua.str.contains("Android", regex=False),
            ua.str.contains("Chrome", regex=False),
            ua.str.contains("Safari", regex=False),
        ],
        ["headless", "avito_app", "yandex", "firefox", "ios_web", "android_web", "chrome", "safari"],
        "other",
    )
    version = (
        ua.str.extract(r"(?:Chrome|Firefox|Avito|Version)/(\d+)", expand=False).astype(float)
    )
    return pd.DataFrame({"ua_family": family, "ua_version": version}, index=ua.index)


def events_in_window(events: pd.DataFrame, meta: pd.DataFrame) -> pd.DataFrame:
    ev = events.merge(meta[["cookie_id", "window_start_ts", "window_end_ts"]], on="cookie_id")
    ev = ev[(ev.event_ts >= ev.window_start_ts) & (ev.event_ts < ev.window_end_ts)]
    return ev.drop(columns=["window_start_ts", "window_end_ts"])


def _entropy(counts: pd.Series) -> float:
    p = counts / counts.sum()
    return float(-(p * np.log(p)).sum())


def item_popularity_features(ev: pd.DataFrame, lookback_days: int = POP_LOOKBACK_DAYS) -> pd.DataFrame:
    #Насколько «общие» объявления смотрит кука.
    #Несколько сервисов-парсеров обходят одни и те же объявления, поэтому у ботов объявления чаще совпадают с другими куками, чем у людей.
    #Для каждого объявления куки считаю, сколько других кук смотрели его за последние 7 дней, включая текущий. Будущие дни и разметку (бот/человек) не использую.
    it = ev.dropna(subset=["item_id"])[["cookie_id", "item_id", "event_ts"]].copy()
    it["day"] = it.event_ts.dt.normalize()
    it = it.drop_duplicates(["cookie_id", "item_id"])
    daily = it.groupby(["item_id", "day"]).size().rename("c").reset_index()

    # объём кол-ва кук с просмотрами за день и за скользящее окно
    cookies_day = it.groupby("day").cookie_id.nunique()
    cookies_day = cookies_day.reindex(pd.date_range(cookies_day.index.min(), cookies_day.index.max()), fill_value=0)
    cookies_trail = cookies_day.rolling(lookback_days, min_periods=1).sum()

    pairs = it[["cookie_id", "item_id", "day"]].merge(daily.rename(columns={"day": "d2"}), on="item_id")
    lag = (pairs.day - pairs.d2).dt.days
    pairs["c_same"] = pairs.c.where(lag == 0, 0)
    pairs = pairs[(lag >= 0) & (lag < lookback_days)]
    agg = pairs.groupby(["cookie_id", "item_id", "day"]).agg(c_trail=("c", "sum"), c_same=("c_same", "sum"))
    agg[["c_trail", "c_same"]] -= 1  # сама кука не считается
    # нормируем на 1000 кук, иначе признак дрейфует вместе с объёмом трафика и длиной истории
    day_idx = agg.index.get_level_values("day")
    agg["c_trail"] = agg.c_trail / cookies_trail.reindex(day_idx).values * 1000
    agg["c_same"] = agg.c_same / cookies_day.reindex(day_idx).values * 1000

    # совместные просмотры с конкретными другими куками за то же скользящее окно:
    # куки одного сервиса делят между собой сразу несколько объявлений
    cc = it[["cookie_id", "item_id", "day"]].merge(it[["cookie_id", "item_id", "day"]], on="item_id")
    cc_lag = (cc.day_x - cc.day_y).dt.days
    cc = cc[(cc.cookie_id_x != cc.cookie_id_y) & (cc_lag >= 0) & (cc_lag < lookback_days)]
    shared = cc.groupby(["cookie_id_x", "cookie_id_y"]).size().groupby(level=0)

    g = agg.groupby("cookie_id")
    return pd.DataFrame({
        "max_items_shared_with_one_cookie": shared.max(),
        "n_cookies_sharing_2plus_items": shared.agg(lambda s: (s >= 2).sum()),
        "item_pop_trail_mean": g.c_trail.mean(),
        "item_pop_trail_max": g.c_trail.max(),
        "item_shared_trail_frac": g.c_trail.agg(lambda c: (c > 0).mean()),
        "item_pop_same_day_mean": g.c_same.mean(),
        "item_shared_same_day_frac": g.c_same.agg(lambda c: (c > 0).mean()),
    })


def build_features(events: pd.DataFrame, meta: pd.DataFrame) -> pd.DataFrame:
    #Возвращает таблицу признаков: одна строка на каждую куку, в том же порядке, что в meta
    #В meta нужно передавать все куки сразу (train и test): популярность объявлений считается по просмотрам всех кук. бот/человек при этом не используется

    ev = events_in_window(events, meta)

    # Некоторые события записаны в данных дважды
    # Сколько таких повторов у куки, сохраняю как признак, а сами повторы удаляю
    dup_mask = ev.duplicated()
    n_dup = dup_mask.groupby(ev.cookie_id).sum().rename("n_dup_rows")
    ev = ev[~dup_mask].copy()

    # Порядок строк в файле произвольный, сортируем по времени
    ev = ev.sort_values(["cookie_id", "event_ts"], kind="mergesort").reset_index(drop=True)
    ev["platform"] = normalize_platform(ev.platform)
    ev = ev.join(parse_user_agent(ev.user_agent))
    g = ev.groupby("cookie_id", sort=False)

    feats = [n_dup]
    n = g.size().rename("n_events")
    feats.append(n)

    # состав событий
    cnt = pd.crosstab(ev.cookie_id, ev.event_name).reindex(columns=EVENT_TYPES, fill_value=0)
    frac = cnt.div(cnt.sum(axis=1), axis=0).add_prefix("frac_")
    cnt = cnt.add_prefix("cnt_")
    feats += [cnt, frac]
    iv = cnt["cnt_item_view"].clip(lower=1)
    funnel = pd.DataFrame({
        "photo_per_item": cnt["cnt_photo_swipe"] / iv,
        "seller_per_item": cnt["cnt_seller_page_view"] / iv,
        "contact_per_item": cnt[[f"cnt_{c}" for c in CONTACT_TYPES]].sum(axis=1) / iv,
        "fav_per_item": cnt["cnt_favorite_add"] / iv,
        "item_per_search": cnt["cnt_item_view"] / cnt["cnt_search_results_view"].clip(lower=1),
    })
    feats.append(funnel)
    feats.append(g.event_name.agg(lambda s: _entropy(s.value_counts())).rename("event_entropy"))

    # время
    ev["dt"] = g.event_ts.diff().dt.total_seconds()
    gd = ev.dropna(subset=["dt"]).groupby("cookie_id").dt
    log_dt = np.log1p(ev["dt"]).groupby(ev.cookie_id)
    timing = pd.DataFrame({
        "dt_min": gd.min(),
        "dt_q10": gd.quantile(0.1),
        "dt_median": gd.median(),
        "dt_mean": gd.mean(),
        "dt_q90": gd.quantile(0.9),
        "dt_std": gd.std(),
        "logdt_std": log_dt.std(),
        "logdt_mean": log_dt.mean(),
        "dt_lt2_frac": gd.agg(lambda x: (x < 2).mean()),
        "dt_lt10_frac": gd.agg(lambda x: (x < 10).mean()),
        # доля самого частого интервала у скриптов с фиксированным sleep
        "dt_mode_frac": gd.agg(lambda x: x.value_counts().iloc[0] / len(x)),
        "dt_nunique_ratio": gd.agg(lambda x: x.nunique() / len(x)),
    })
    timing["dt_cv"] = timing.dt_std / timing.dt_mean.replace(0, np.nan)
    # Похожа ли каждая пауза на предыдущую. У людей паузы идут сериями, у ботов соседние паузы не связаны
    timing["logdt_autocorr"] = (
        ev.assign(ldt=np.log1p(ev.dt), ldt_prev=np.log1p(g.dt.shift()))
        .dropna(subset=["ldt", "ldt_prev"])
        .groupby("cookie_id")
        .apply(lambda d: d.ldt.corr(d.ldt_prev) if len(d) > 2 else np.nan)
    )
    feats.append(timing)

    ev["new_session"] = ev.dt.isna() | (ev.dt > SESSION_GAP_S)
    ev["hour"] = ev.event_ts.dt.hour
    first_ts, last_ts = g.event_ts.min(), g.event_ts.max()
    span = (last_ts - first_ts).dt.total_seconds()
    n_sessions = g.new_session.sum()
    feats.append(pd.DataFrame({
        "span_s": span,
        "events_per_hour_active": n / (span / 3600).clip(lower=1 / 60),
        "n_sessions": n_sessions,
        "events_per_session": n / n_sessions,
        "hours_active": g.hour.nunique(),
        "night_frac": g.hour.agg(lambda h: h.between(0, 5).mean()),
        "hour_first": g.hour.first(),
        "hour_mean": g.hour.mean(),
        "second_zero_frac": g.event_ts.agg(lambda t: (t.dt.second == 0).mean()),
        "minute_unique_ratio": g.event_ts.agg(lambda t: t.dt.floor("min").nunique() / len(t)),
    }))

    # разнообразие контента
    items = ev.dropna(subset=["item_id"]).groupby("cookie_id").item_id
    div = pd.DataFrame({
        "item_nunique": g.item_id.nunique(),
        "item_revisit_frac": items.agg(lambda x: 1 - x.nunique() / len(x)),
        "cat_nunique": g.item_category.nunique(),
        "cat_top_share": g.item_category.agg(
            lambda x: x.value_counts(normalize=True).iloc[0] if x.notna().any() else np.nan),
        "cat_entropy": g.item_category.agg(
            lambda x: _entropy(x.value_counts()) if x.notna().any() else np.nan),
        "loc_nunique": g.item_location.nunique(),
        "loc_entropy": g.item_location.agg(
            lambda x: _entropy(x.value_counts()) if x.notna().any() else np.nan),
        "pro_seller_frac": g.seller_type.agg(
            lambda x: (x == "pro").mean() if x.notna().any() else np.nan),
    })
    div["loc_per_item"] = div.loc_nunique / div.item_nunique.clip(lower=1)
    # как часто соседние события уходят в другой город/категорию
    for col, name in [("item_location", "loc_switch_frac"), ("item_category", "cat_switch_frac")]:
        known = ev.dropna(subset=[col])
        prev = known.groupby("cookie_id")[col].shift()
        div[name] = (known[col] != prev)[prev.notna()].groupby(known.cookie_id).mean()
    feats.append(div)
    feats.append(item_popularity_features(ev))

    # время на странице до следующего события по типу текущего события
    ev["next_dt"] = -g.event_ts.diff(-1).dt.total_seconds()
    for et, name in [("item_view", "dwell_item"), ("search_results_view", "dwell_search"),
                     ("photo_swipe", "dwell_photo")]:
        feats.append(ev[ev.event_name == et].groupby("cookie_id").next_dt.median().rename(name))

    # поиск
    s = ev[ev.event_name == "search_results_view"]
    gs = s.groupby("cookie_id")
    search = pd.DataFrame({
        "query_nunique": gs.search_query.nunique(),
        "page_max": gs.search_page.max(),
        "page_mean": gs.search_page.mean(),
        "page_gt1_frac": gs.search_page.agg(lambda x: (x > 1).mean()),
        # листание выдачи строго подряд 1,2,3,... внутри одного запроса
        "page_step1_frac": gs.apply(lambda d: ((d.search_page.diff() == 1)
                                               & (d.search_query == d.search_query.shift())).mean()),
        "query_latin_frac": gs.search_query.agg(
            lambda q: q.dropna().str.fullmatch(r"[a-z0-9 .\-]+").mean() if q.notna().any() else np.nan),
    })
    search["pages_per_query"] = gs.size() / search.query_nunique.clip(lower=1)
    feats.append(search)

    # курсор (есть только у web)
    web = ev[ev.platform == "web"].copy()
    gw = web.groupby("cookie_id")
    web["ptr_step"] = np.hypot(gw.pointer_x.diff(), gw.pointer_y.diff())
    gw = web.groupby("cookie_id")
    pointer = pd.DataFrame({
        "ptr_frac": gw.pointer_x.agg(lambda x: x.notna().mean()),
        "ptr_x_std": gw.pointer_x.std(),
        "ptr_y_std": gw.pointer_y.std(),
        "ptr_x_mean": gw.pointer_x.mean(),
        "ptr_y_mean": gw.pointer_y.mean(),
        "ptr_x_range": gw.pointer_x.max() - gw.pointer_x.min(),
        "ptr_y_range": gw.pointer_y.max() - gw.pointer_y.min(),
        "ptr_step_median": gw.ptr_step.median(),
        "ptr_step_zero_frac": gw.ptr_step.agg(lambda x: (x.dropna() == 0).mean() if x.notna().any() else np.nan),
        "ptr_edge_frac": gw.apply(lambda d: ((d.pointer_x <= 5) | (d.pointer_y <= 5)).sum()
                                  / max(d.pointer_x.notna().sum(), 1)),
    })
    pointer["ptr_xy_corr"] = gw.apply(
        lambda d: d.pointer_x.corr(d.pointer_y) if d.pointer_x.notna().sum() > 2 else np.nan)
    feats.append(pointer)

    # клиент
    client = pd.DataFrame({
        "platform": g.platform.agg(lambda x: x.mode().iloc[0]),
        "ua_family": g.ua_family.agg(lambda x: x.mode().iloc[0]),
        "ua_version": g.ua_version.max(),
        "ua_nunique": g.user_agent.nunique(),
        "platform_nunique": g.platform.nunique(),
    })
    feats.append(client)

    f = pd.concat(feats, axis=1)
    f.index.name = "cookie_id"

    # мета признаки куки
    m = meta.set_index("cookie_id")
    out = m[[]].join(f)
    out["cookie_age_days"] = (m.window_start_ts - m.cookie_created_at).dt.total_seconds() / 86400
    out["created_in_window"] = (m.cookie_created_at >= m.window_start_ts).astype(int)
    out["first_event_after_created_h"] = (first_ts.reindex(out.index) - m.cookie_created_at).dt.total_seconds() / 3600
    out["first_event_hour_of_window"] = (first_ts.reindex(out.index) - m.window_start_ts).dt.total_seconds() / 3600

    for c in ["platform", "ua_family"]:
        out[c] = out[c].astype("category")
    zero_if_missing = ("cnt_", "frac_", "n_events", "n_dup_rows", "max_items_shared", "n_cookies_sharing")
    out = out.fillna({c: 0 for c in out.columns if c.startswith(zero_if_missing)})
    return out.reset_index()
