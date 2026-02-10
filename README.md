# SkillRL: Evolving Agents via Recursive Skill-Augmented Reinforcement Learning

<div align="center">

Bridging the gap between raw experience and policy improvement through automatic skill discovery.

</div>

<p align="center">
<img src="figs/pipeline.png" width="80%" alt="SKILLRL Pipeline Overview">
</p>

## 🔥 News

- **[02/10/2026]** SkillRL paper was released on [arXiv](https://arxiv.org/abs/2602.08234)!

## 📖 Overview

SkillRL is a framework that enables LLM agents to learn high-level, reusable behavioral patterns from past experiences. While traditional memory-based methods store redundant and noisy raw trajectories, SKILLRL abstracts these into a hierarchical skill library.

## 🤖 Key Features

- **Experience-based Skill Distillation**: Transforms successful trajectories into strategic patterns and failed ones into concise lessons from failure. 

- **Hierarchical SKILLBANK**: Organizes knowledge into General Skills for universal strategic guidance and Task-Specific Skills for category-level heuristics. 

- **Recursive Skill Evolution**: A dynamic mechanism where the skill library co-evolves with the agent's policy during RL by analyzing validation failures.

- **Context Efficiency**: Achieves 10-20x token compression compared to raw trajectory storage while enhancing reasoning utility. 

## 🚀 Getting Started (Coming Soon)

We are currently preparing the codebase for public release.

## 📚 Citation
If you find our work helpful, please consider citing:

```bibtex
@article{xia2026skillrl,
  title={SKILLRL: Evolving Agents via Recursive Skill-Augmented Reinforcement Learning},
  author={Xia, Peng and Chen, Jianwen and Wang, Hanyang and Liu, Jiaqi and Zeng, Kaide and Wang, Yu and Han, Siwei and Zhou, Yiyang and Zhao, Xujiang and Zhao, Haifeng and Zheng, Zeyu and Xie, Cihang and Yao, Huaxiu},
  journal={arXiv preprint arXiv:2602.08234},
  year={2026}
}
```