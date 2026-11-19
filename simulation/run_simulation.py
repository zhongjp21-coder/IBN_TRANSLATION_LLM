from mininet.net import Mininet
from mininet.node import Controller, OVSSwitch
from mininet.log import setLogLevel, info
import time

def mininet_feasibility_demo():
    setLogLevel('info')
    info("\n*** 1. 启动 Mininet 仿真拓扑 (h1 -- s1 -- h2)\n")
    # 使用自带的最轻量级控制器，仅用于激活网络
    net = Mininet(controller=Controller, switch=OVSSwitch)
    
    c0 = net.addController('c0')
    s1 = net.addSwitch('s1')
    h1 = net.addHost('h1', ip='10.0.0.1/24')
    h2 = net.addHost('h2', ip='10.0.0.2/24')
    
    net.addLink(h1, s1)
    net.addLink(h2, s1)
    
    net.start()
    time.sleep(2) # 等待网络收敛
    
    # ---------------------------------------------------------
    # Stage A: 初始状态基线测试 (S_initial)
    # ---------------------------------------------------------
    info("\n*** 2. [S_initial] 注入初始测试流量 (h1 ping h2)...\n")
    ping_res1 = h1.cmd('ping -c 3 10.0.0.2')
    if "0% packet loss" in ping_res1:
        info("    -> [基线正常] 初始状态下 h1 与 h2 互通。\n")
    
    # ---------------------------------------------------------
    # Stage B: 模拟下发 LLM 翻译的网络意图
    # ---------------------------------------------------------
    info("\n*** 3. 模拟下发 LLM 意图: [Security Intent] 阻断 h1 到 h2\n")
    # 在真实系统中，这是 LLM 翻译出的 JSON 并发给 SDN 控制器。
    # 为了验证仿真本身的可行性，我们直接通过 ovs-ofctl 工具注入底层流表。
    s1.cmd('ovs-ofctl add-flow s1 priority=100,ip,nw_src=10.0.0.1,nw_dst=10.0.0.2,actions=drop')
    info("    -> [部署成功] 阻断流表已注入底层数据平面 (s1 交换机)。\n")
    
    # ---------------------------------------------------------
    # Stage C: 闭环保障验证 (S_final & Assertion)
    # ---------------------------------------------------------
    info("\n*** 4. [S_final] 闭环保障验证 (Assurance) - 再次打流测试...\n")
    ping_res2 = h1.cmd('ping -c 3 10.0.0.2')
    
    # 提取丢包率进行断言
    if "100% packet loss" in ping_res2:
        info("\n✅ [验证成功/Intent Satisfied] 意图生效！\n")
        info("    测试结果：数据平面按预期丢弃了报文，网络行为与 LLM 意图完全对齐。\n")
    else:
        info("\n❌ [验证失败/Intent Failed] 意图未生效，发生意图偏移 (Intent Drift)。\n")
        
    net.stop()

if __name__ == '__main__':
    mininet_feasibility_demo()