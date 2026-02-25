import socket
import json
import time

def recv_data(sock, length):
    data = b''
    while len(data) < length:
        chunk = sock.recv(min(4096, length - len(data)))
        if not chunk:
            break
        data += chunk
    return data.decode('utf-8')

def handle_client(client_socket, handle_request):
    try:
        data_length = int(recv_data(client_socket, 9))
        data = recv_data(client_socket, data_length)

        if not data:
            # print("客户端断开连接")
            return -1
        try:
            request = json.loads(data)
            # print(f"收到客户端请求: {request}")
            response = handle_request(request)
            if response is None:
                return 666
            response_data = json.dumps(response, ensure_ascii=False).encode('utf-8')
            response_length = f"{len(response_data):9d}".encode('utf-8')
            client_socket.send(response_length + response_data)
            return 0
        except json.JSONDecodeError:
            print("接收到的数据不是有效的JSON格式")
            return -1
    except Exception as e:
        print(e)
        return -1
    finally:
        client_socket.close()

def start_server(handle_request, host='::', port=61111):
    while True:
        try:
            server_socket = socket.socket(socket.AF_INET6, socket.SOCK_STREAM)
            server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            server_socket.bind((host, port))
            server_socket.listen(200)
            print(f"服务端启动，正在监听 {host}:{port}...")
            break
        except Exception as e:
            print(e)
            time.sleep(5)
    
    while True:
        client_socket, client_address = server_socket.accept()
        # print(f"客户端 {client_address} 已连接")
        ret = handle_client(client_socket, handle_request)
        if ret == 666:
            break

def send_request(server_ip, server_port, request={}):
    client_socket = socket.socket(socket.AF_INET6, socket.SOCK_STREAM)
    try:
        client_socket.connect((server_ip, server_port))
        request_data = json.dumps(request, ensure_ascii=False).encode('utf-8')
        request_length = f"{len(request_data):9d}".encode('utf-8')
        client_socket.send(request_length + request_data)

        data_length = int(recv_data(client_socket, 9))
        data = recv_data(client_socket, data_length)
        # print(data)
        response = json.loads(data)
        return response
        # print(f"收到服务端响应: {response}")
    except Exception as e:
        # print(e)
        return None
    finally:
        client_socket.close()
