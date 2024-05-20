---
title: Konfigurasi Load Balancing dengan Nginx dan HAProxy di Ubuntu 24
author: fdciabdul
type: post
date: 2024-04-20T13:20:01+00:00
categories:
  - Tutorial
---

# Apa Itu Load Balancing?

Load balancing itu adalah teknik buat ngebagi beban traffic ke beberapa server biar performa dan ketersediaan website tetap terjaga. Ada beberapa jenis load balancer, tapi di sini kita fokus ke dua yang populer: Nginx dan HAProxy.

## Persiapan

Sebelum mulai, pastikan kita punya:
- Server Ubuntu 24
- Akses root atau sudo
- Beberapa server backend buat ngetes load balancing

### Update Sistem

Pertama, update dulu sistem kita biar semua paket terbaru terinstall.

```sh
sudo apt update
sudo apt upgrade -y
```

## Konfigurasi Nginx sebagai Load Balancer

Nginx itu nggak cuma web server, tapi juga bisa jadi load balancer yang powerful. Yuk, kita mulai dengan instalasi dan konfigurasinya.

### Instalasi Nginx

Install Nginx dengan perintah ini:

```sh
sudo apt install nginx -y
```

### Konfigurasi Nginx

Setelah Nginx terinstall, kita perlu konfigurasi file Nginx buat load balancing. Buka file konfigurasi default Nginx.

```sh
sudo nano /etc/nginx/sites-available/default
```

Tambahkan konfigurasi load balancing di dalamnya:

```nginx
upstream backend_servers {
    server backend1.example.com;
    server backend2.example.com;
    server backend3.example.com;
}

server {
    listen 80;

    location / {
        proxy_pass http://backend_servers;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }
}
```

Di sini, kita bikin sebuah upstream bernama `backend_servers` yang isinya list server backend kita. Lalu, kita arahkan semua traffic ke upstream tersebut.

### Restart Nginx

Setelah konfigurasi selesai, simpan file dan keluar dari editor. Lalu restart Nginx biar perubahan diterapkan.

```sh
sudo systemctl restart nginx
```

## Konfigurasi HAProxy sebagai Load Balancer

HAProxy juga load balancer yang hebat dan sering dipake buat aplikasi yang butuh kecepatan tinggi. Berikut cara instalasi dan konfigurasinya.

### Instalasi HAProxy

Install HAProxy dengan perintah ini:

```sh
sudo apt install haproxy -y
```

### Konfigurasi HAProxy

Buka file konfigurasi HAProxy:

```sh
sudo nano /etc/haproxy/haproxy.cfg
```

Edit file tersebut dengan konfigurasi berikut:

```haproxy
global
    log /dev/log    local0
    log /dev/log    local1 notice
    chroot /var/lib/haproxy
    stats socket /run/haproxy/admin.sock mode 660 level admin
    stats timeout 30s
    user haproxy
    group haproxy
    daemon

defaults
    log     global
    mode    http
    option  httplog
    option  dontlognull
    timeout connect 5000
    timeout client  50000
    timeout server  50000
    errorfile 400 /etc/haproxy/errors/400.http
    errorfile 403 /etc/haproxy/errors/403.http
    errorfile 408 /etc/haproxy/errors/408.http
    errorfile 500 /etc/haproxy/errors/500.http
    errorfile 502 /etc/haproxy/errors/502.http
    errorfile 503 /etc/haproxy/errors/503.http
    errorfile 504 /etc/haproxy/errors/504.http

frontend http_front
    bind *:80
    default_backend http_back

backend http_back
    balance roundrobin
    server backend1 backend1.example.com:80 check
    server backend2 backend2.example.com:80 check
    server backend3 backend3.example.com:80 check
```

Di sini, kita konfigurasi frontend buat dengerin di port 80 dan arahkan traffic ke backend yang diatur pake round-robin.

### Restart HAProxy

Setelah konfigurasi selesai, simpan file dan keluar dari editor. Lalu restart HAProxy biar perubahan diterapkan.

```sh
sudo systemctl restart haproxy
```

## Testing Load Balancer

Setelah konfigurasi selesai, saatnya kita tes apakah load balancer kita udah berfungsi dengan baik.

### Uji Coba Nginx

Buka browser dan akses alamat IP atau domain server load balancer kita. Kalau konfigurasi benar, kita bakal lihat halaman dari salah satu backend server kita.

### Uji Coba HAProxy

Lakukan hal yang sama seperti di atas, akses alamat IP atau domain server HAProxy kita. Seharusnya kita bakal diarahkan ke salah satu backend server kita juga.

## Monitoring dan Maintenance

Buat monitoring, kita bisa pake beberapa tools tambahan kayak Grafana, Prometheus, atau langsung dari log yang disimpan oleh Nginx dan HAProxy. Jangan lupa buat rutin ngecek dan maintenance server-server kita biar tetap optimal.

## Kesimpulan

Load balancing itu penting banget buat nge-handle traffic yang tinggi dan memastikan website atau aplikasi kita tetap cepat dan tersedia. Dengan Nginx dan HAProxy, kita bisa bikin load balancer yang kuat dan reliable di Ubuntu 24. Jangan ragu buat eksplorasi lebih lanjut dan sesuaikan konfigurasi sesuai kebutuhan kita.

Semoga artikel ini membantu, dan selamat mencoba!
