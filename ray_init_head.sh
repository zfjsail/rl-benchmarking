# Head节点（第一个节点）：
bashray start --head --port=6379 && sleep infinity
# Worker节点（第二个节点）：
bashray start --address=<head_node_ip>:6379 && sleep infinity