import base64
import hmac
import requests
import time
import hashlib
from urllib.parse import urlencode

api_key = '61b50738254fe40001d497f5'
api_secret = '6ab6a43e-41f6-45f0-a6fc-6b926da40d40'
api_passphrase = 'fish_taco'

def get_all_balances():
    # Request & clean account balance information for all used accounts
    now = int(time.time() * 1000)
    url_endpoint = '/api/v1/accounts'
    url = 'https://api.kucoin.com/api/v1/accounts'
    str_to_sign = str(now) + 'GET' + url_endpoint
    signature = base64.b64encode(
    hmac.new(api_secret.encode('utf-8'), str_to_sign.encode('utf-8'), hashlib.sha256).digest())
    passphrase = base64.b64encode(hmac.new(api_secret.encode('utf-8'), api_passphrase.encode('utf-8'), hashlib.sha256).digest())
    headers = {
        "KC-API-SIGN": signature,
        "KC-API-TIMESTAMP": str(now),
        "KC-API-KEY": api_key,
        "KC-API-PASSPHRASE": passphrase,
        "KC-API-KEY-VERSION": "2"
    }
    response = requests.request('get', url, headers=headers)
    data = response.json()["data"]
    data_clean = []
    for d in data:
        crncy = d["currency"]
        acc_type = d["type"]
        account = crncy+"_"+acc_type
        bal = d["available"]
        data_clean.append({account: bal})
    return data_clean

def get_balance(crncy):
    # Request & clean account balance information for a currency
    now = int(time.time() * 1000)
    url_endpoint = '/api/v1/accounts'
    url = 'https://api.kucoin.com/api/v1/accounts'
    str_to_sign = str(now) + 'GET' + url_endpoint + "?currency=" + crncy
    signature = base64.b64encode(
    hmac.new(api_secret.encode('utf-8'), str_to_sign.encode('utf-8'), hashlib.sha256).digest())
    passphrase = base64.b64encode(hmac.new(api_secret.encode('utf-8'), api_passphrase.encode('utf-8'), hashlib.sha256).digest())
    headers = {
        "KC-API-SIGN": signature,
        "KC-API-TIMESTAMP": str(now),
        "KC-API-KEY": api_key,
        "KC-API-PASSPHRASE": passphrase,
        "KC-API-KEY-VERSION": "2"
    }
    response = requests.request('get', url + "?currency=" + crncy, headers=headers)
    data = response.json()["data"]
    data_clean = []
    for d in data:
        acc_type = d["type"]
        account = crncy+"_"+acc_type
        bal = d["available"]
        data_clean.append({account: bal})
    return data_clean

def get_history(symbol):
    # Request & clean data for the last 100 trades of a symbol
    now = int(time.time() * 1000)
    url_endpoint = '/api/v1/market/histories'
    url = 'https://api.kucoin.com/api/v1/market/histories'
    str_to_sign = str(now) + 'GET' + url_endpoint + "?symbol=" + symbol
    signature = base64.b64encode(
    hmac.new(api_secret.encode('utf-8'), str_to_sign.encode('utf-8'), hashlib.sha256).digest())
    passphrase = base64.b64encode(hmac.new(api_secret.encode('utf-8'), api_passphrase.encode('utf-8'), hashlib.sha256).digest())
    headers = {
        "KC-API-SIGN": signature,
        "KC-API-TIMESTAMP": str(now),
        "KC-API-KEY": api_key,
        "KC-API-PASSPHRASE": passphrase,
        "KC-API-KEY-VERSION": "2"
    }
    response = requests.request('get', url + "?symbol=" + symbol, headers=headers)
    data = response.json()["data"]
    return data

def get_candles(symbol, type, startAt, endAt):
    # Request & clean candle data between unix timestamps startAt & endAt. Max 1500 candles.
    now = int(time.time() * 1000)
    url_endpoint = '/api/v1/market/candles'
    url = 'https://api.kucoin.com/api/v1/market/candles'
    str_to_sign = str(now) + 'GET' + url_endpoint + "?type=" + type + "&symbol=" + symbol + "&startAt=" + startAt + "&endAt=" + endAt
    signature = base64.b64encode(
    hmac.new(api_secret.encode('utf-8'), str_to_sign.encode('utf-8'), hashlib.sha256).digest())
    passphrase = base64.b64encode(hmac.new(api_secret.encode('utf-8'), api_passphrase.encode('utf-8'), hashlib.sha256).digest())
    headers = {
        "KC-API-SIGN": signature,
        "KC-API-TIMESTAMP": str(now),
        "KC-API-KEY": api_key,
        "KC-API-PASSPHRASE": passphrase,
        "KC-API-KEY-VERSION": "2"
    }
    response = requests.request('get', url + "?type=" + type + "&symbol=" + symbol + "&startAt=" + startAt + "&endAt=" + endAt, headers=headers)
    data = response.json()
    return data

def get_margin_acc():
    # Request & clean margin account data
    now = int(time.time() * 1000)
    url_endpoint = '/api/v1/margin/account'
    url = 'https://api.kucoin.com/api/v1/margin/account'
    str_to_sign = str(now) + 'GET' + url_endpoint
    signature = base64.b64encode(
    hmac.new(api_secret.encode('utf-8'), str_to_sign.encode('utf-8'), hashlib.sha256).digest())
    passphrase = base64.b64encode(hmac.new(api_secret.encode('utf-8'), api_passphrase.encode('utf-8'), hashlib.sha256).digest())
    headers = {
        "KC-API-SIGN": signature,
        "KC-API-TIMESTAMP": str(now),
        "KC-API-KEY": api_key,
        "KC-API-PASSPHRASE": passphrase,
        "KC-API-KEY-VERSION": "2"
    }
    response = requests.request('get', url, headers=headers)
    data = response.json()["data"]
    debt_ratio = data["debtRatio"]
    accounts = data["accounts"]
    data_clean = {"debtRatio": debt_ratio}
    for a in accounts:
        crncy = a["currency"]
        del a["currency"]
        data_clean[crncy] = a
    return data_clean

def get_debt_ratio():
    # Request & clean margin account data
    now = int(time.time() * 1000)
    url_endpoint = '/api/v1/margin/account'
    url = 'https://api.kucoin.com/api/v1/margin/account'
    str_to_sign = str(now) + 'GET' + url_endpoint
    signature = base64.b64encode(
    hmac.new(api_secret.encode('utf-8'), str_to_sign.encode('utf-8'), hashlib.sha256).digest())
    passphrase = base64.b64encode(hmac.new(api_secret.encode('utf-8'), api_passphrase.encode('utf-8'), hashlib.sha256).digest())
    headers = {
        "KC-API-SIGN": signature,
        "KC-API-TIMESTAMP": str(now),
        "KC-API-KEY": api_key,
        "KC-API-PASSPHRASE": passphrase,
        "KC-API-KEY-VERSION": "2"
    }
    response = requests.request('get', url, headers=headers)
    data = response.json()["data"]
    data_clean = {"debtRatio": data["debtRatio"]}
    return data_clean


def test_sell_mrg(symbol, funds):
    now = int(time.time() * 1000)
    url_endpoint = '/api/v1/margin/order'
    url = 'https://api.kucoin.com/api/v1/margin/order'
    message = urlencode({
        "clientOid": "zvupg"+str(now),
        "symbol": symbol,
        "side": "sell",
        "type": "market",
        "funds": str(funds)
    })
    str_to_sign = str(now) + 'POST' + url_endpoint + "?" + message
    signature = base64.b64encode(
        hmac.new(api_secret.encode('utf-8'), str_to_sign.encode('utf-8'), hashlib.sha256).digest())
    passphrase = base64.b64encode(
        hmac.new(api_secret.encode('utf-8'), api_passphrase.encode('utf-8'), hashlib.sha256).digest())
    headers = {
        "KC-API-SIGN": signature,
        "KC-API-TIMESTAMP": str(now),
        "KC-API-KEY": api_key,
        "KC-API-PASSPHRASE": passphrase,
        "KC-API-KEY-VERSION": "2"
    }
    response = requests.request('post', url + "?" + message, headers=headers)
    return response.json()