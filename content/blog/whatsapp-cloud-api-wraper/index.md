---
title: WhatsApp Cloud API Wrapper
author: fdciabdul
type: post
date: 2024-03-11T17:11:58+00:00
url: /whatsapp-cloud-api-wrapper/
featured_image: http://imtaqin.id/wp-content/uploads/2024/03/image-2.png
alia_post_layout:
  - fullwidth
hits:
  - 87
categories:
  - Project

---
WhatsApp Cloud API adalah versi WhatsApp API yang dihosting di cloud, dan juga dikenal sebagai WhatsApp Business Platform. Versi ini dihosting di server cloud Meta.

WhatsApp Cloud API memungkinkan bisnis menengah dan besar untuk berinteraksi dengan pelanggan menggunakan versi yang ada di-host di cloud Meta. WhatsApp Cloud API menawarkan banyak fitur yang memberdayakan bisnis untuk terhubung dan berinteraksi dengan pelanggan.

**Library Whatsapp Cloud API**  
Sebenarnya begitu banyak library untuk whatsapp cloud api ini, namun saya belum menemukan library yang include parsing untuk notifikasi event yang masuk didalam webhooknya, untuk itu saya membuat Library ini  
  
Untuk cara penginstallan cukuo mudah. anda hanya perlu NodeJS sebagai runtime nya, dan install dependency ini .

```
npm install wacloudapi
```

untuk menggunakan library ini, tambahkan library ini di  _require_ atau  _import_  statementnya.

```
const { Message, WAParser, WebhookServer } = require('wacloudapi);
```

lalu beberapa function bisa dilihat dibawah ini  
  

### Sending Messages

Create a new `Message` instance with your API credentials:

```
const message = new Message(apiVersion, phoneNumberId, accessToken);
```

You can now use the various methods provided by the _`Message`_ class to send messages:message.sendTextMessage(recipientPhoneNumber, messageContent);

### Webhook Server

Create a new `WebhookServer` instance and specify the desired port and whether to use ngrok:

```
const webhookServer = new WebhookServer(port, useNgrok, ngrokAuthToken);
```

add a listener for incoming messages:

```
webhookServer.on('message', (message) => { console.log('Received message:', message); });
```

Add a route for webhook verification:

```
webhookServer.Verification(callbackUrl, verificationToken);

Start the webhook server:webhookServer.start();
```

## Webhook Parser

The `WAParser` class is used to parse incoming webhook data from the WhatsApp Business API.

#### parseMessage()

Returns the parsed message object depending on the type of message contained in the received webhook data.

```
const parse = new WAParser(WebhookData); // parse message const parsedMessage = parse.parseMessage();
```

## Notification Parser

The `NotificationParser` class is used to parse incoming webhook data from the WhatsApp Business API.

#### NotificationParser()

Returns the parsed message object depending on the type of message contained in the received webhook data.

```
const parse = new NotificationParser(WebhookData); // parse message const parsedMessage = parse.parseNotification();
```

SHARE