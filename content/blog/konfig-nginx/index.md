---
title: 🔁 Konfigurasi Reverse Proxy Nginx untuk Situs Cloudflare
summary: Reverse Proxy adalah mekanisme yang memungkinkan Anda mengarahkan permintaan pengguna dari server web Nginx Anda ke server web lain.
date: 2024-05-12
authors:
  - fdciabdul
tags:
  - DevOps
  - Linux
---


Reverse Proxy adalah mekanisme yang memungkinkan Anda mengarahkan permintaan pengguna dari server web Nginx Anda ke server web lain. Dalam artikel ini, kami akan membahas cara melakukan konfigurasi Nginx sebagai Reverse Proxy dengan Cloudflare. Konfigurasi ini bermanfaat untuk meningkatkan keamanan dan performa situs web Anda.

## Struktur Konfigurasi Nginx Reverse Proxy

Berikut adalah contoh konfigurasi Nginx yang digunakan sebagai Reverse Proxy:

```
server {
    listen 80;
    listen 443 ssl;
    server_name _; 
    index index.html index.htm;
    root /var/www/html/; 
    client_max_body_size 100M;

    # Konfigurasi SSL
    ssl_certificate /path/ker/certtificate.crt;
    ssl_certificate_key /root/cert/private.key;
    ssl_protocols TLSv1.2 TLSv1.3;
    ssl_prefer_server_ciphers on;
    ssl_ciphers 'EECDH+AESGCM:EDH+AESGCM:AES256+EECDH:AES256+EDH';
    ssl_session_cache shared:SSL:10m;
    ssl_session_timeout 10m;
    ssl_stapling on;
    ssl_stapling_verify on;

    # Mengizinkan IP Cloudflare
    set_real_ip_from 173.245.48.0/20;
    set_real_ip_from 103.21.244.0/22;
    set_real_ip_from 103.22.200.0/22;
    set_real_ip_from 103.31.4.0/22;
    set_real_ip_from 141.101.64.0/18;
    set_real_ip_from 108.162.192.0/18;
    set_real_ip_from 190.93.240.0/20;
    set_real_ip_from 188.114.96.0/20;
    set_real_ip_from 197.234.240.0/22;
    set_real_ip_from 198.41.128.0/17;
    set_real_ip_from 162.158.0.0/15;
    set_real_ip_from 104.16.0.0/13;
    set_real_ip_from 104.24.0.0/14;
    set_real_ip_from 172.64.0.0/13;
    set_real_ip_from 131.0.72.0/22;

    # Mengizinkan IP Cloudflare (IPv6)
    set_real_ip_from 2400:cb00::/32;
    set_real_ip_from 2606:4700::/32;
    set_real_ip_from 2803:f800::/32;
    set_real_ip_from 2405:b500::/32;
    set_real_ip_from 2405:8100::/32;
    set_real_ip_from 2a06:98c0::/29;
    set_real_ip_from 2c0f:f248::/32;

    # Menggunakan header Cloudflare untuk IP asli user
    real_ip_header CF-Connecting-IP;

    # Konfigurasi resolver untuk DNS
    resolver 1.1.1.1 8.8.8.8 valid=300s ipv6=off;

    location / {
        proxy_pass https://www.brandw0w135.com;
        proxy_set_header Host www.brandw0w135.com; 
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        proxy_redirect off;
        proxy_ssl_server_name on;
        proxy_ssl_verify off;
        proxy_ssl_protocols TLSv1.2 TLSv1.3;
        proxy_ssl_ciphers 'EECDH+AESGCM:EDH+AESGCM:AES256+EECDH:AES256+EDH';
        proxy_connect_timeout 90;
        proxy_send_timeout 90;
        proxy_read_timeout 90;
    }
}

```

## Penjelasan Konfigurasi

Mari kita bahas bagian-bagian dari konfigurasi ini secara detail:

### 1. Blok  `server`

-   `listen 80;`  dan  `listen 443 ssl;`: Mendengarkan permintaan pada port HTTP (80) dan HTTPS (443).
-   `server_name _;`: Mengatur nama domain yang akan dilayani oleh server ini.
-   `index index.html index.htm;`: Menentukan file indeks yang akan dicari jika tidak ada file yang ditentukan.
-   `root /var/www/html/;`: Menentukan direktori root untuk file statis.
-   `client_max_body_size 100M;`: Menentukan batas ukuran unggahan file hingga 100MB.

### 2. Konfigurasi SSL

-   `ssl_certificate`  dan  `ssl_certificate_key`: Menentukan lokasi sertifikat SSL dan kunci privat.
-   `ssl_protocols TLSv1.2 TLSv1.3;`: Mengizinkan hanya protokol SSL yang aman.
-   `ssl_prefer_server_ciphers on;`: Mengizinkan server memilih cipher.
-   `ssl_ciphers`: Daftar cipher yang digunakan untuk enkripsi.
-   `ssl_session_cache shared:SSL:10m;`: Menyimpan sesi SSL hingga 10MB.
-   `ssl_session_timeout 10m;`: Menentukan waktu kedaluwarsa sesi SSL selama 10 menit.
-   `ssl_stapling on;`  dan  `ssl_stapling_verify on;`: Mengaktifkan OCSP stapling untuk verifikasi sertifikat SSL.

### 3. Mengizinkan IP Cloudflare

-   `set_real_ip_from`: Menentukan rentang IP Cloudflare untuk mengizinkan akses.
-   `real_ip_header CF-Connecting-IP;`: Menggunakan header Cloudflare  `CF-Connecting-IP`  untuk mendeteksi IP asli pengguna.

### 4. Resolver DNS

-   `resolver 1.1.1.1 8.8.8.8`: Mengatur resolver DNS dengan Cloudflare dan Google DNS.
-   `valid=300s ipv6=off;`: Menentukan validitas resolver selama 300 detik dan menonaktifkan IPv6.

### 5. Blok  `location`

-   `proxy_pass https://www.brandw0w135.com;`: Meneruskan permintaan ke server lain.
-   `proxy_set_header`: Menambahkan header tambahan saat meneruskan permintaan.
-   `proxy_redirect off;`: Menonaktifkan pengalihan otomatis.
-   `proxy_ssl_server_name on;`: Mengaktifkan penggunaan nama server untuk SSL.
-   `proxy_ssl_verify off;`: Menonaktifkan verifikasi sertifikat SSL server tujuan.
-   `proxy_ssl_protocols`  dan  `proxy_ssl_ciphers`: Mengatur protokol SSL dan cipher untuk permintaan yang diteruskan.
-   `proxy_connect_timeout 90;`,  `proxy_send_timeout 90;`, dan  `proxy_read_timeout 90;`: Menentukan batas waktu koneksi, pengiriman, dan pembacaan.