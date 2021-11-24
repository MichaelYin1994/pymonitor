## **基于Pyhton的轻量级Linux系统资源监控工具**

---
### **项目说明**

基于Python的轻量级系统GPU、CPU、系统内存、网络与磁盘监控工具。运行在基于NVIDIA官方Container构建的Python镜像内。通过psutil工具与pynvml工具搜集系统各种资源信息，并存储到本地sqlite数据库中。

可以配置即时通讯软件（例如钉钉或者飞书）的API_URL，从而实现发送系统级警告到远程系统管理员的功能。

---
### **系统环境与主要依赖packages**

- python == 3.6.9
- pynvml == 11.0.0
- psutil == 5.8.0
- numpy == 1.19.5
- scikit-learn == 0.24.2
- pandas == 1.1.5
