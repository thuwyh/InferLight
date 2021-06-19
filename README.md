## InferLight

Tools for batch inference.

### Test result
#### Bert inference
- Single text inference.

```
Concurrency Level:      32
Time taken for tests:   10.164 seconds
Complete requests:      1000
Failed requests:        0
Total transferred:      202000 bytes
HTML transferred:       111000 bytes
Requests per second:    98.39 [#/sec] (mean)
Time per request:       325.234 [ms] (mean)
Time per request:       10.164 [ms] (mean, across all concurrent requests)
Transfer rate:          19.41 [Kbytes/sec] received
```

- Batch inference
```
Concurrency Level:      32
Time taken for tests:   4.019 seconds
Complete requests:      1000
Failed requests:        999
   (Connect: 0, Receive: 0, Length: 999, Exceptions: 0)
Total transferred:      202978 bytes
HTML transferred:       111978 bytes
Requests per second:    248.79 [#/sec] (mean)
Time per request:       128.620 [ms] (mean)
Time per request:       4.019 [ms] (mean, across all concurrent requests)
Transfer rate:          49.32 [Kbytes/sec] received
```