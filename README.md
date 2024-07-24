# BUPT 2024 Spring - Machine Learning - Course Project

[English Version README](#en)

## 赛道二：日志异常检测

数据集为某系统运行过程中所产生的日志信息，为了实现系统自动报警能力，需要实现一个机器学习算法，利用日志信息判断系统是否有异常发生。我们收集到了多条日志信息，并对其进行了标注，标注分为两类：`Normal` 和 `Anomalous`。

例如：

```json
{
  "log": "1135434878 2005.12.24 R67-M1-NC-C:J04-U01 RAS KERNEL r16=0xfffffffe r17=0x4c00081f r18=0x44000000 r19=0x085f9780",
  "label": "Normal"
}
```

我们使用当下流行的大语言模型来讲解决日志是否异常这样一个文本二分类问题。基于 `bert-base-cased` 对其使用我们的数据集进行微调,
得到可以完成指定日志是否异常的二分类任务。

使用 Huggingface 提供的 `transformers` 、 `datasets` 、 `evaluate` 库实现预训练模型的加载、微调和评估。在 200 条数据上测试最终可以得到较高的正确率。

<h2 id="en">Track 2: Log Anomaly Detection</h2>

The data set is the log information generated during the operation of a system. In order to realize the automatic alarm capability of the system, it is necessary to implement a machine learning algorithm to use the log information to determine whether the system has anomalies. We collected multiple log information and annotated them. The annotations are divided into two categories: `Normal` and `Anomalous`.

For example:

```json
{
  "log": "1135434878 2005.12.24 R67-M1-NC-C:J04-U01 RAS KERNEL r16=0xfffffffe r17=0x4c00081f r18=0x44000000 r19=0x085f9780",
  "label": "Normal"
}
```

We use the currently popular large language model to solve the text binary classification problem of whether the log is abnormal. Based on `bert-base-cased`, we fine-tune it with our dataset, and get a binary classification task that can complete whether the specified log is abnormal.

Use the `transformers`, `datasets`, and `evaluate` libraries provided by Huggingface to load, fine-tune, and evaluate the pre-trained model. Testing on 200 data finally yields a high accuracy rate.
