from web3 import Web3
from ipfshttpclient import Client
import json

# 连接到以太坊节点
w3 = Web3(Web3.HTTPProvider('http://localhost:8545'))  # 你需要替换成你自己的以太坊节点地址

# 加载智能合约ABI
with open('HashStorage.json', 'r') as f:
    abi = json.load(f)

# 合约地址
contract_address = '0x1234567890123456789012345678901234567890'  # 你需要替换成你自己的合约地址

# 加载合约
contract = w3.eth.contract(address=contract_address, abi=abi)

# 连接到IPFS节点
ipfs = Client('127.0.0.1', 5001)  # 你需要根据你的IPFS节点配置修改地址

# 上传模型到IPFS并获取哈希值
with open('trained_model.pth', 'rb') as f:
    res = ipfs.add(f)
    ipfs_hash = res['Hash']

# 发送交易将IPFS哈希值存入区块链
tx_hash = contract.functions.storeHash(ipfs_hash).transact()

# 等待交易确认
receipt = w3.eth.wait_for_transaction_receipt(tx_hash)

print("IPFS哈希值存入区块链成功，交易哈希为:", tx_hash.hex())
