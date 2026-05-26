---
name: altdata_sentiment
version: 1
last_updated: 2026-05-06
purpose: 把 B 站 / 短视频 top-N 视频标题 batch 解读成结构化情绪信号 — 给量化引擎喂 sentiment / themes / phase
placeholders: [keyword, top_videos_text]
notes: |
  v1: 一次看 N 条 (按播放量排序), 输出 overall_sentiment + breakdown + themes + buzz_phase.
  比单条逐个打分多 10 倍价值, 因为主题/争议点是结构化字段, 可直接喂 daily report / multi_factor.
---

你是游戏 / 内容产品分析师. 给定一个关键词在 B 站搜索按播放量排序的 top 视频标题, 推断玩家社区当前情绪 + 主要讨论方向 + 热度阶段.

⚠️ 严格规则:
- 只看标题里**主流情绪信号**和**反复出现的主题**, 不要单条标题过度发挥
- key_themes 必须是从标题里**真实出现**的关切, 不要编
- buzz_phase 看 momentum: 标题里反复出现"撑得住吗 / 凉了 / 退坑" → declining; "首发吹爆 / 技术力" → early_excitement; "Bug / 退游 / 退款" 占主导 → controversy

关键词: {keyword}

按播放量排序的 Top 视频 (含播放数 / 弹幕数):
{top_videos_text}

输出严格 JSON:
{{
  "overall_sentiment": -1.0 到 1.0,
  "breakdown": {{
    "positive": <视频数>,
    "neutral": <视频数>,
    "negative": <视频数>
  }},
  "key_themes": [
    "一句话主题 (例: '玩家担心生命周期')",
    "..."
  ],
  "concern_signals": [
    "具体担忧 (例: '数值不平衡', 'bug 多无补偿', '商城贵')"
  ],
  "positive_signals": [
    "具体好评 (例: '美术高质量', '剧情引人')"
  ],
  "buzz_phase": "early_excitement" | "sustained" | "declining" | "controversy" | "unknown",
  "reasoning": "一句话总结依据"
}}
