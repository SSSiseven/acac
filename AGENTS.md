# Codex Working Rules

## Project goal

在ACAC官方代码基础上，实现并验证TG-ACAC：
使用动态任务图表示Overcooked中的任务实例，通过机器人—任务注意力、
任务依赖传播和可执行性掩码，为ready智能体选择任务节点，同时保留
ACAC的agent-centric history encoder、异步GAE和PPO训练机制。

## Scope

当前只处理RQ1的公开环境快速验证。

第一阶段环境：
- ovcA7：调试
- ovcAR7：快速主验证
- ovcAR11_N6：规模验证

当前不实现：
- UAV—四足自建搜索环境
- LLM、VLM、世界模型
- ROS/Gazebo
- ACAC异步GAE的重新设计
- 底层导航或宏动作执行器重写

## Modification rules

1. 原始ACAC必须保持可运行。
2. 不直接覆盖原始算法类；新增tg_acac入口。
3. 环境动力学、奖励和宏动作执行器保持不变。
4. 所有行为变化必须通过配置控制。
5. 先写单元测试，再接入训练。
6. 正式训练前必须完成smoke test。
7. 不修改结果CSV，不手工美化实验结果。
8. 不删除用户已有修改。
9. 不执行git push。
10. 可以创建本地checkpoint提交，但必须说明改动内容。

## Experimental fairness

ACAC与TG-ACAC必须使用：
- 相同地图
- 相同随机种子
- 相同训练episode数
- 相同evaluation设置
- 相同奖励
- 相同宏动作执行器
- 相同异步GAE和PPO实现

需要增加参数量匹配的ACAC-MLP对照。

## Required workflow

1. 审计仓库和依赖；
2. 复现ACAC smoke test；
3. 输出代码调用链；
4. 设计TaskGraphObservation；
5. 实现纯函数任务图构造器；
6. 编写任务图单元测试；
7. 实现TG-ACAC-NoDependency；
8. 实现完整TG-ACAC；
9. 完成三种子pilot；
10. 满足退出条件后再运行正式实验。

## Reporting

每完成一个阶段，更新：
- STATUS.md
- DECISIONS.md
- REPRODUCIBILITY.md

报告必须区分：
- 已完成
- 已验证
- 待验证
- 当前阻塞