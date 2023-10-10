import dipu_ext.ext_


# 这里定义一个python层的组合算子实现
def native_m_nms():
    pass


#  判断是否存在pybind的函数，如果存在，则调用；否则调用python层的组合算子
try:
    m_nms = dipu_ext.ext_.m_nms
except:
    m_nms = native_m_nms
