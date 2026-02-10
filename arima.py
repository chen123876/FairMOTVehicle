import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pymysql
from sqlalchemy import create_engine
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error
import warnings
warnings.filterwarnings('ignore')

# 设置中文显示
plt.rcParams["font.family"] = ["SimHei", "WenQuanYi Micro Hei", "Heiti TC"]
plt.rcParams["axes.unicode_minus"] = False

# 数据库配置
DB_CONFIG = {
    'host': 'localhost',
    'port': 3306,
    'user': 'root',
    'password': '021228',
    'db': 'weibo',
    'charset': 'utf8mb4'
}

# 1. 连接数据库并读取数据
def load_data():
    # 创建数据库连接引擎
    engine = create_engine(
        f"mysql+pymysql://{DB_CONFIG['user']}:{DB_CONFIG['password']}@"
        f"{DB_CONFIG['host']}:{DB_CONFIG['port']}/{DB_CONFIG['db']}?"
        f"charset={DB_CONFIG['charset']}"
    )
    
    # 读取数据表（注意：原时间列名是date，不是data，已修正）
    query = """
        SELECT date, total_comments 
        FROM daily_comment_stats 
        ORDER BY date ASC
    """
    df = pd.read_sql(query, engine)
    
    # 数据预处理
    df['date'] = pd.to_datetime(df['date'])  # 转换为日期格式
    df.set_index('date', inplace=True)       # 设置日期为索引
    df = df.asfreq('D')                     # 确保按天频率
    
    # 处理缺失值
    if df.isnull().any().any():
        df['total_comments'] = df['total_comments'].interpolate(method='time')
        print("提示：数据存在缺失值，已通过时间插值法填充")
    
    print(f"数据加载完成：共{len(df)}条记录，时间范围：{df.index.min()}至{df.index.max()}")
    return df

# 2. 数据可视化与训练集测试集划分
def prepare_data(data):
    # 绘制整体评论量趋势图
    plt.figure(figsize=(12, 6))
    plt.plot(data.index, data['total_comments'], 'b-', alpha=0.7)
    plt.title('微博每日评论量时间序列')
    plt.xlabel('日期')
    plt.ylabel('评论数')
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig('微博评论量整体趋势.png', dpi=300)
    plt.close()
    
    # 按7:3比例划分训练集和测试集（时间顺序）
    split_idx = int(len(data) * 0.7)
    train = data.iloc[:split_idx]
    test = data.iloc[split_idx:]
    
    # 绘制划分结果
    plt.figure(figsize=(12, 6))
    plt.plot(train.index, train['total_comments'], 'b-', label='训练集')
    plt.plot(test.index, test['total_comments'], 'r-', label='测试集')
    plt.axvline(x=train.index[-1], color='gray', linestyle='--', alpha=0.5)
    plt.title('训练集与测试集划分')
    plt.xlabel('日期')
    plt.ylabel('评论数')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig('训练集测试集划分.png', dpi=300)
    plt.close()
    
    return train, test

# 3. 平稳性检验与差分处理
def check_stationarity(train_series):
    # ADF检验
    result = adfuller(train_series)
    adf_stat = result[0]
    p_value = result[1]
    critical_values = result[4]
    
    print(f"\nADF平稳性检验结果：")
    print(f"ADF统计量: {adf_stat:.4f}")
    print(f"p值: {p_value:.4f}")
    print(f"临界值: {critical_values}")
    
    # 判断是否平稳
    is_stationary = p_value <= 0.05
    print(f"结论：{'序列平稳' if is_stationary else '序列非平稳，需要差分处理'}")
    
    d = 0
    diff_series = train_series.copy()
    
    # 若不平稳，进行1阶差分
    if not is_stationary:
        d = 1
        diff_series = train_series.diff().dropna()
        # 差分后再次检验
        diff_result = adfuller(diff_series)
        print(f"\n1阶差分后ADF检验p值: {diff_result[1]:.4f}")
        
        # 绘制差分前后对比图
        plt.figure(figsize=(14, 6))
        plt.subplot(121)
        plt.plot(train_series.index, train_series, 'b-')
        plt.title('原始序列')
        plt.grid(alpha=0.3)
        
        plt.subplot(122)
        plt.plot(diff_series.index, diff_series, 'g-')
        plt.title('1阶差分后序列')
        plt.grid(alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('序列差分对比.png', dpi=300)
        plt.close()
    
    return d, diff_series

# 4. ACF与PACF分析确定p和q
def determine_pq(diff_series, d):
    # 绘制ACF和PACF图
    plt.figure(figsize=(14, 6))
    plt.subplot(121)
    plot_acf(diff_series, lags=20, ax=plt.gca())
    plt.title('自相关图(ACF)')
    plt.grid(alpha=0.3)
    
    plt.subplot(122)
    plot_pacf(diff_series, lags=20, ax=plt.gca())
    plt.title('偏自相关图(PACF)')
    plt.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('ACF_PACF分析.png', dpi=300)
    plt.close()
    
    # 网格搜索最优p和q
    best_aic = float('inf')
    best_pq = (0, 0)
    p_range = range(0, 4)
    q_range = range(0, 4)
    
    print("\n开始网格搜索最优ARIMA参数...")
    for p, q in [(p, q) for p in p_range for q in q_range]:
        try:
            model = ARIMA(diff_series, order=(p, d, q))
            results = model.fit()
            if results.aic < best_aic:
                best_aic = results.aic
                best_pq = (p, q)
            print(f"ARIMA({p},{d},{q}) - AIC: {results.aic:.2f}")
        except:
            continue
    
    print(f"最优参数: ARIMA{best_pq + (d,)}，AIC值: {best_aic:.2f}")
    return best_pq[0], best_pq[1]

# 5. 模型拟合与预测
def arima_forecast(train, test, p, d, q):
    # 拟合ARIMA模型
    model = ARIMA(train['total_comments'], order=(p, d, q))
    results = model.fit()
    
    # 输出模型摘要
    print("\n模型参数摘要:")
    print(results.summary().tables[1])
    
    # 训练集拟合
    train_pred = results.fittedvalues
    
    # 测试集预测
    test_pred = results.get_forecast(steps=len(test))
    test_pred_values = test_pred.predicted_mean
    test_pred_ci = test_pred.conf_int()
    
    # 计算误差
    mse = mean_squared_error(test['total_comments'], test_pred_values)
    rmse = np.sqrt(mse)
    print(f"\n测试集预测RMSE: {rmse:.2f}")
    
    # 绘制拟合与预测结果
    plt.figure(figsize=(14, 7))
    plt.plot(train.index, train['total_comments'], 'b-', label='训练集实际值')
    plt.plot(train.index[d:], train_pred[d:], 'g--', label='训练集拟合值')  # 跳过差分损失的前d个值
    plt.plot(test.index, test['total_comments'], 'r-', label='测试集实际值')
    plt.plot(test.index, test_pred_values, 'm--', label='测试集预测值')
    plt.fill_between(test.index, 
                    test_pred_ci.iloc[:, 0], 
                    test_pred_ci.iloc[:, 1], 
                    color='pink', alpha=0.3, label='95%置信区间')
    
    plt.axvline(x=train.index[-1], color='gray', linestyle='--', alpha=0.5)
    plt.title(f'ARIMA({p},{d},{q})模型拟合与预测结果')
    plt.xlabel('日期')
    plt.ylabel('评论数')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig('ARIMA预测结果.png', dpi=300)
    plt.close()
    
    # 绘制残差图
    residuals = results.resid[d:]  # 跳过前d个残差
    plt.figure(figsize=(12, 6))
    plt.subplot(121)
    residuals.plot()
    plt.axhline(y=0, color='r', linestyle='--')
    plt.title('残差序列')
    
    plt.subplot(122)
    residuals.plot(kind='kde')
    plt.title('残差密度分布')
    
    plt.tight_layout()
    plt.savefig('模型残差分析.png', dpi=300)
    plt.close()
    
    return results

# 6. 未来7天预测
def forecast_future(model, data, days=7):
    # 预测未来7天
    future_forecast = model.get_forecast(steps=days)
    future_values = future_forecast.predicted_mean
    future_ci = future_forecast.conf_int()
    
    # 绘制未来预测图
    plt.figure(figsize=(12, 6))
    plt.plot(data.index[-30:], data['total_comments'].iloc[-30:], 'b-', label='最近30天实际值')
    plt.plot(future_values.index, future_values, 'r--', label=f'未来{days}天预测值')
    plt.fill_between(future_values.index,
                    future_ci.iloc[:, 0],
                    future_ci.iloc[:, 1],
                    color='pink', alpha=0.3, label='95%置信区间')
    
    plt.title(f'微博评论量未来{days}天预测')
    plt.xlabel('日期')
    plt.ylabel('评论数')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig('未来评论量预测.png', dpi=300)
    plt.close()
    
    # 输出未来预测结果
    print("\n未来7天评论量预测结果:")
    future_df = pd.DataFrame({
        '日期': future_values.index.strftime('%Y-%m-%d'),
        '预测评论数': round(future_values).astype(int),
        '下限': round(future_ci.iloc[:, 0]).astype(int),
        '上限': round(future_ci.iloc[:, 1]).astype(int)
    })
    print(future_df.to_string(index=False))

# 主函数
def main():
    # 步骤1: 加载数据
    data = load_data()
    if len(data) < 30:
        print("数据量不足（至少需要30条记录），无法进行有效建模")
        return
    
    # 步骤2: 数据准备与划分
    train, test = prepare_data(data)
    
    # 步骤3: 平稳性检验
    d, diff_series = check_stationarity(train['total_comments'])
    
    # 步骤4: 确定ARIMA参数
    p, q = determine_pq(diff_series, d)
    
    # 步骤5: 模型拟合与测试集预测
    model = arima_forecast(train, test, p, d, q)
    
    # 步骤6: 未来7天预测
    forecast_future(model, data, days=7)

if __name__ == "__main__":
    main()
    