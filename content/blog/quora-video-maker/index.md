---
title: Quora Video Maker
author: fdciabdul
type: post
date: 2024-04-12T13:20:01+00:00
url: /quora-video-maker/

hits:
  - 52
categories:
  - Project

---
Proyek ini berupa sebuah aplikasi yang mampu mengubah konten dari Quora menjadi video. Aplikasi ini saya rancang dengan tujuan untuk membantu teman-teman yang ingin memanfaatkan konten berkualitas dari Quora dan mengubahnya menjadi format video yang lebih menarik dan mudah dicerna.

Berikut ini adalah penjelasan singkat tentang cara kerja aplikasi ini.

  1. **Memasukkan URL Quora**: Pertama-tama, aplikasi ini akan meminta kepada teman-teman untuk memasukkan URL dari thread Quora yang ingin diubah menjadi video.
  2. **Ekstraksi Konten**: Setelah URL diberikan, aplikasi ini akan melakukan proses ekstraksi konten. Ekstraksi ini mencakup judul thread, konten dari postingan utama, dan juga balasan dari pengguna lain pada thread tersebut.
  3. **Konversi Teks ke Audio**: Selanjutnya, aplikasi ini akan melakukan konversi dari teks ke audio. Proses ini melibatkan dua layanan berbeda, yaitu OpenAI dan TikTok TTS. Judul dan konten postingan akan dikonversi menggunakan OpenAI, sedangkan balasan dari pengguna lain akan dikonversi menggunakan TikTok TTS.
  4. **Pembuatan Video**: Setelah semua audio berhasil dihasilkan, aplikasi ini akan memproses semua audio tersebut dan menggabungkannya menjadi satu video. Video ini nantinya akan mencakup judul, konten postingan, dan juga balasan dari pengguna lain.
  5. **Penyimpanan**: Akhirnya, video yang dihasilkan akan disimpan dengan nama yang berasal dari judul thread Quora yang telah dibersihkan dari karakter-karakter yang tidak diinginkan.

Proyek ini saya bangun dengan menggunakan berbagai library dan modul yang ada di Node.js, seperti OpenAI untuk konversi teks ke audio, Puppeteer untuk ekstraksi konten dari Quora, dan juga Fluent-ffmpeg untuk pembuatan video. Saya juga menggunakan beberapa library lainnya untuk membantu dalam proses pembuatan aplikasi ini.

  
Demo :

{{< youtube zptVgzeFB1E>}}

|Source :|Author  |Author  |
|--|--|--|
| Projects.co.id | [**azharrattan**](https://projects.co.id/public/browse_users/view/355213/azharrattan-azharrattan) |  [https://projects.co.id/public/past_projects/view/42af1b/quora-to-tiktok-video-generator](https://projects.co.id/public/past_projects/view/42af1b/quora-to-tiktok-video-generator)|


&nbsp;