"""Send raw signals + recommendations to LLM, get back compact action list.

Routed through quant.llm_router so it picks up the same provider/model chain
as newswatch / investigator (see config/llm_routes.yaml). The legacy
config/llm.yaml endpoint is kept only as a manual override for ad-hoc tests.
"""
from __future__ import annotations
import json
import logging
from typing import Any

from . import config as cfg_mod
from . import llm_router

log = logging.getLogger(__name__)

SYSTEM_PROMPT = """\
你是一个量化操作单格式化器。把输入的量化分析结果 (JSON) 格式化为带"证据 + 信心 + 反向警示"的中文操作单。

严格规则:
1. 不要发挥/预测/给 JSON 中没有的信息
2. 不要寒暄, 不要总结性结尾
3. 价格、金额、股数、权重必须来自 JSON, 不要估算或四舍五入
4. 金额是 delta_value (正=买入, 负=卖出); 币种看 currency ($ 或 ¥)
5. 股数是 delta_shares: 美股 4 位小数; A 股 整数 100 股一手
6. action 已确定性裁决, 只能格式化不可改: ADD / WATCH_BUY / REDUCE / WATCH_SKIP / HOLD / STOP_LOSS / DEFER_TO_LLM

**conviction 渲染 (★ 数 = JSON 的 conviction 字段 0-5)**:
- 0/5 → 空白不显示
- 1/5 → ★☆☆☆☆
- 2/5 → ★★☆☆☆
- 3/5 → ★★★☆☆
- 4/5 → ★★★★☆
- 5/5 → ★★★★★

**top_factors 渲染规则 (核心!)**:
- top_factors 是 list of dict, 每条有 name, score, contribution, evidence
- name 字段做中文翻译: technical=技术 | events=事件 | trade_signals=订单流 | sentiment=情绪 |
  fundamental=基本面 | analyst=分析师 | momentum=动量 | macro_regime=宏观 |
  alt_data=领先指标 | rating_change=评级变化 | event_intensity=事件烈度
- 每条 top_factors 渲染一行 bullet: `• {中文 name} ({contribution:+.2f}): {evidence}`
- 至多渲染 3 条 (按 |contribution| 大小)
- evidence 已是中文人话, 直接照用; 不要再翻译

**counter_factors 渲染规则**:
- 如果非空, 加 "⚠️ 反向因子" 子段, 每条同上格式
- 至多 2 条
- 没 counter_factors 这段不显示

**展示名规则**:
- 美股 (currency=USD): 直接用 symbol (VOO/AMD/SOXX)
- A 股 (currency=CNY): 用 display_name "中文名 (六位代号)", 如 `完美世界 (002624)`. 绝对不写 `002624.SZ`

输出格式 (Markdown, Telegram 渲染):

📊 *YYYY-MM-DD 操作单*
📅 *数据截至 YYYY-MM-DD 收盘* (freshness.max_stale_days > 1 时尾加 ⚠️ 数据 N 天前)

如 earnings_this_week / important_dates 任一非空, 加段:
📣 *本周大事*
  📊 财报: `SYMBOL` (持仓/关注) 今天⚡/X天后 EPS预期$Y
  💰 除权: `SYMBOL` X天后 ($Y/股)
  🌍 宏观: REGION FOMC/CPI/NFP X天后 HH:MM UTC

🇺🇸 美股: $XXXX 当日 +X.XX%   (如有 USD 持仓)
🇨🇳 A股: ¥XXXX 当日 +X.XX%   (如有 CNY 持仓)
_盘后/盘前价请发 ad-hoc 问_

**操作分组 (按 action 字段)**:

🟢 *加仓* (action=ADD)
• `<显示名>` 加 <币种><金额> (≈<股数> 股) | 现价 <币种>X.XX | 信心 ★★★☆☆ (3/5)
  📊 关键因子:
    • 事件 (+0.16): 财报已出超预期 +5.5%, 6d 前
    • 订单流 (+0.11): Call/Put OI 极偏多 (PC ratio 0.00)
    • 评级变化 (+0.02): 目标价 +9.7%
  ⚠️ 反向: (如有)
    • 技术 (-0.05): RSI 超买 73
  权重: X.X% → Y.Y%

🆕 *关注买入* (action=WATCH_BUY, 关注池触发或持仓加分较弱)
同上格式. 来自 watchlist 项目用"关注", 来自 positions 用"加仓"。

🔴 *减仓* (action=REDUCE) / ⚠️ *止损* (action=STOP_LOSS)
同上格式; 信心 ★ 仍要展示。

🤖 *待 LLM 综合判断* (action=DEFER_TO_LLM)
• `<symbol>` 信心 ★★★☆☆ (3/5) — catalyst_imminent=true, 单维度信号不可靠
  📊 关键因子: <列前 3 条>
  💡 建议: 主人发"<symbol> 现在能不能加" ad-hoc, 让 deepseek 综合判断

✅ *持有不动* (action=HOLD): SYM1, SYM2, ...
🚫 *暂无信号* (action=WATCH_SKIP): SYM1, SYM2, ... 共 N 只 conviction <2/5

**重要**: 全是 HOLD/WATCH_SKIP 时, 不要输出"今日无操作建议, 全部持有"。改输出:
📊 *本日无新信号*: <N> 只持仓全部中性 (avg conviction <2/5)
🔍 关键持仓 conviction 分布: AMD (2/5), NVDA (3/5), 002624 (1/5)
_说明: 多因子综合中性 ≠ 不动, 而是缺乏明确 entry/exit 触发点。如有 ad-hoc 问题随时调阿雷分析。_
⚖️ *风险提示*: <如有>

格式细节:
- 金额单位严格按 currency: USD → $, CNY → ¥
- 美股股数 4 位小数 (Acorns 类零碎股); A 股股数取整且必须凑整百, 写"≈300 股"
- 不同币种的标的不要互相比较权重 (各算各的 100%)
- 当 abs(delta_value) < (USD: 5 / CNY: 50) 时, 视为不操作, 归入持有不动
- 如果 report_date 与 data_date 相同, 省略 "(数据日期 ...)"

如果完全没有 ADD/REDUCE/STOP_LOSS/WATCH_BUY 操作:
   只输出: "📊 YYYY-MM-DD: 今日无操作建议, 全部持有"

**基本面参考**: 输入 JSON 里有 `fundamentals_hints` 数组 (PE/PB/历史分位等). 在每条减仓/加仓建议**后追加一行 缩进 4 空格** "估值: <fundamentals_hints[symbol].summary>"
仅当对应股有 fundamentals_hints 才加这行, 没有就跳过, 不要发挥。

**昨晚音频要点 (overnight_audio)**: 如果输入 JSON 里 overnight_audio 数组非空, 在最后加一段:

🎙️ *昨晚音频要点 ({N} 段)*
• [来源] 标题截断至60字 (重要级 X/10)
  💡 一句话总结 (取 summary 前 80 字)
  影响: SYMBOL 方向 ±X% (如有 impacts)

每段 1-3 行。如果 overnight_audio 为空, 不加这段。

不要有其他文字。
"""


def package(raw_output: dict, *, llm_cfg: dict | None = None) -> str:
    # Compact the input to save tokens — drop verbose fields LLM doesn't need
    compact = _compact(raw_output)

    # Route via llm_router (task="format" → glm-5.1 first then kimi-k2.6,
    # per config/llm_routes.yaml). Falls through the whole chain if any
    # provider 401/timeouts.
    result = llm_router.chat(
        prompt=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": json.dumps(compact, ensure_ascii=False)},
        ],
        task="format",
        max_tokens=4096,
        temperature=0.2,
        timeout=300,
        disable_thinking=True,  # output is plain markdown; don't burn budget on thinking field
    )
    text = (result.get("text") or "").strip()
    backend = result.get("backend") or "?"
    tokens = result.get("tokens") or {}
    log.info("LLM packager via %s, tokens=%s/%s",
              backend, tokens.get("prompt", "?"), tokens.get("completion", "?"))
    return text


def _compact(raw: dict) -> dict:
    out: dict[str, Any] = {
        "report_date": raw["generated_at"][:10],
        "report_time_utc": raw["generated_at"],
        "data_date": raw.get("data_date"),
        "freshness": raw.get("freshness", {}),
        "by_currency": raw["portfolio"].get("by_currency", {}),
        "positions": [],
        "watchlist": [],
        "risk_notes": raw.get("risk_notes", []),
        "fundamentals_hints": [],   # 基本面摘要给 LLM 参考
        "overnight_audio": raw.get("overnight_audio", []),
        "earnings_this_week": [
            {
                "symbol": e["symbol"],
                "name": e.get("name"),
                "is_held": e.get("is_held"),
                "report_date": e["report_date"],
                "days_until": e["days_until"],
                "eps_estimate": e.get("eps_estimate"),
            }
            for e in raw.get("earnings_this_week", [])
        ],
        "important_dates": {
            "corporate": [
                {"symbol": c["symbol"], "type": c["event_type"], "date": c["event_date"],
                 "amount": c.get("amount")}
                for c in raw.get("important_dates", {}).get("corporate", [])
            ],
            "macro": [
                {"type": m["event_type"], "region": m["region"], "date": m["event_date"],
                 "time_utc": m.get("event_time_utc")}
                for m in raw.get("important_dates", {}).get("macro", [])
            ],
        },
    }
    fundamentals_data = raw.get("fundamentals", {})
    for sym, f in fundamentals_data.items():
        if not f:
            continue
        extra = f.get("extra") or {}
        bits = []
        if f.get("pe"):
            pct = extra.get("pe_pct_5y")
            bits.append(f"PE {f['pe']:.1f}" + (f" (历史 {pct*100:.0f} 分位)" if pct else ""))
        if f.get("pb"):
            bits.append(f"PB {f['pb']:.1f}")
        if f.get("revenue_yoy") is not None:
            bits.append(f"营收同比 {f['revenue_yoy']*100:+.1f}%")
        if extra.get("forward_pe"):
            bits.append(f"前瞻PE {extra['forward_pe']:.1f}")
        # 分析师 upside (合并自 analyst_ratings, 入 extra.analyst_ratings)
        ratings = extra.get("analyst_ratings") or {}
        if ratings.get("upside_pct") is not None:
            ups = ratings["upside_pct"]
            bits.append(f"卖方目标 ${ratings.get('target_mean_price','?')} (upside {ups:+.1f}%)")
        if bits:
            out["fundamentals_hints"].append({"symbol": sym, "summary": " | ".join(bits)})

    weights = raw["portfolio"]["weights"]
    currencies = raw["portfolio"].get("currencies", {})
    sigs = raw["signals"]

    for rec in raw["recommendations"]:
        sym = rec["symbol"]
        sig = sigs.get(sym, {})
        currency = rec.get("currency") or currencies.get(sym, "USD")
        notes = rec.get("notes") or {}
        item = {
            "symbol": sym,
            "display_name": rec.get("display_name") or sym,
            "currency": currency,
            "action": rec["action"],
            "current_weight_pct": round(rec["current_weight"] * 100, 1),
            "target_weight_pct": round(rec["target_weight"] * 100, 1),
            "current_value": rec.get("current_value", 0),
            "target_value": rec.get("target_value", 0),
            "delta_value": rec.get("delta_value", 0),
            "delta_shares": rec.get("delta_shares", 0),
            "reasons": rec["reason_codes"],
            "confidence": rec["confidence"],
            "conviction": notes.get("conviction", 0),                       # 0-5 stars
            "composite_score": notes.get("composite_score"),
            "top_factors": notes.get("top_factors", []),                    # structured list[dict]
            "counter_factors": notes.get("counter_factors", []),            # structured list[dict]
            "top_factor_evidence": notes.get("top_factor_evidence", []),    # flat string list (legacy)
            "factor_scores": notes.get("factor_scores", {}),
            "catalyst_imminent": notes.get("catalyst_imminent", False),
            "factor_breakdown": _compact_factor_breakdown(raw.get("multi_factor", {}).get(sym, {})),
            "price": sig.get("price"),
            "chg_1d_pct": sig.get("chg_1d_pct"),
            "rsi": round(sig.get("rsi", 0), 1),
            "signal_codes": sig.get("signal_codes", []),
        }
        if sym in weights:
            out["positions"].append(item)
        else:
            out["watchlist"].append(item)

    return out


def _compact_factor_breakdown(multi: dict) -> dict:
    breakdown = multi.get("factor_breakdown") or {}
    out: dict[str, Any] = {}
    for name, item in breakdown.items():
        if not isinstance(item, dict):
            continue
        out[name] = {
            "score": item.get("score"),
            "factors": (item.get("factors") or [])[:2],
        }
    return out
