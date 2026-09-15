import os
import logging

class FakeSFTP:
    """
    A fake SFTP client for testing purposes. It simulates the behavior of an SFTP client by storing files in the current directory.
    """

    def __init__(self):
        self.wd = os.path.dirname(__file__)
        logging.info(f"Working directory for fake SFTP: {self.wd}")
        scan_params_path = os.path.join(self.wd, "C:/Lidar/System/Scan parameters/")
        dynscan_path = os.path.join(self.wd, "C:/Users/End User/DynScan/")
        os.makedirs(scan_params_path, exist_ok=True)
        os.makedirs(dynscan_path, exist_ok=True)
        self.files = []
        
    def listdir(self, path): return os.listdir(os.path.join(self.wd, path))
    def get(self, remote, local): 
        remote_path = os.path.join(self.wd, remote)
        with open(remote_path, "rb") as f:
            data = f.read()
        with open(local, "wb") as f:
            f.write(data)
    def put(self, local, remote): 
        logging.info(f"Putting file {local} to {remote} in fake SFTP at {self.wd}.")
        if remote.startswith("/"):
            remote = remote[1:]
        remote_path = os.path.join(self.wd, remote)
        with open(remote_path, "wb") as f:
            f.write(open(local, "rb").read())
        self.files.append(remote_path)
    def close(self):
        for file in self.files:
            os.remove(file)
        logging.info("Removed files from fake SFTP.")
        os.removedirs(os.path.join(self.wd, "C:/Lidar/System/Scan parameters/"))
        logging.info("Removed scan parameters directory from fake SFTP.")
        os.removedirs(os.path.join(self.wd, "C:/Users/End User/DynScan/"))
    def __enter__(self): return self
    def __exit__(self, exc_type, exc_val, exc_tb): self.close()

class FakeSSHClient:
    """
    A fake SSH client for testing purposes. It simulates the behavior of an SSH client by providing a context manager that returns a FakeSFTP instance.
    """

    def __init__(self):
        self.sftp = FakeSFTP()
    def set_missing_host_key_policy(self, policy): pass
    def connect(self, ip_addr, username, password): print(f"Connected to {ip_addr} with username {username}")
    def open_sftp(self): return self.sftp
    def close(self): self.sftp.close()
    def __enter__(self): return self
    def __exit__(self, exc_type, exc_val, exc_tb): self.close()

