### Install Docker

How to fix error c



Source: https://forums.docker.com/t/apt-get-update-err-1-https-download-docker-com-linux-ubuntu-jammy-inrelease-certificate-verification-failed-the-certificate-is-not-trusted/131588/3

```
 kali@kali:~$ sudo apt-get install \
                ca-certificates \
                curl \
                gnupg \
                lsb-release
```

After this, reboot.

```
kali@kali:~$ sudo apt update
kali@kali:~$ sudo apt install -y docker.io
kali@kali:~$ sudo systemctl enable docker --now
kali@kali:~$ docker
kali@kali:~$ sudo usermod -aG docker $USER
```

