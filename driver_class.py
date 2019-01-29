import mxnet as mx
print (mx.__version__)
import math

from custom_service import CustomService

_service = CustomService()

def handle(data, context):
    if not _service.initialized:
        _service.initialize(context)

    if data is None:
        return None

    return _service.handle(data, context)

def percentile(val, arr):
    idx = int(math.ceil((len(arr) - 1) * val / 100.0))
    return arr[idx]

if __name__=="__main__":
    import time

    ctx = mx.cpu(0)

    handle("The joke", ctx)
    times = list()
    for i in range(1, 20):
        start = time.time()
        handle("The joke", ctx)
        end = time.time()
        times.append(end - start)
        print ("i : " + str(i) + " time : " + str(end - start))


    times.sort()

    # print times
    p99 = percentile(99, times) * 1000
    p90 = percentile(90, times) * 1000
    p50 = percentile(50, times) * 1000
    average = sum(times) / len(times) * 1000
    print ("p99 in ms, %f" % p99)
    print ("p90 in ms, %f" % p90)
    print ("p50 in ms, %f" % p50)
    print ("average in ms, %f" % average)
