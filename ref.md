REF:   https://zhuanlan.zhihu.com/p/433074172



# 安装
```
pip install gluonts 
# as gluonts relies on mxnet 
# install MXnet using
pip pip install mxnet
```
# 入门

我们已经看到使用 TensorFlow 和 PyTorch 进行时间序列预测，但它们带有大量代码并且需要对框架非常熟练。 GluonTS 提供用于运行时间序列预测的简单且即时的代码，这里是运行 GluonTS 以使用 DeepAR 预测 Twitter 数量的示例代码。
```
from gluonts.dataset import common
from gluonts.model import deepar
from gluonts.trainer import Trainer
import pandas as pd
```
# 读取数据
```
url = "https://raw.githubusercontent.com/numenta/NAB/master/data/realTweets/Twitter_volume_AMZN.csv"
df = pd.read_csv(url, header=0, index_col=0)
data = common.ListDataset([{
    "start": df.index[0],
    "target": df.value[:"2015-04-05 00:00:00"]
}], freq="5min")
 ```                         
# 初始化deepAR模型
```
trainer = Trainer(epochs=10)
estimator = deepar.DeepAREstimator(
    freq="5min", prediction_length=12, trainer=trainer)
predictor = estimator.train(training_data=data)
```
# 得到预测结果

```
prediction = next(predictor.predict(data))
print(prediction.mean)
prediction.plot(output_file='graph.png')
```
