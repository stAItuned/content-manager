---
title: "KnockKnock: Automate Your Machine Learning Notifications with Ease"
author: Daniele Moltisanti
topics: [AI]
target: Advanced
language: English
cover: cover.webp
meta: "Automate machine learning notifications with KnockKnock, a Python library that integrates with Desktop, Telegram, Email, and Slack. Save time and monitor your training scripts efficiently"
date: 2024-10-11
published: true
---


# KnockKnock: Automate Your Machine Learning Notifications with Ease

## Introduction

Training machine learning models can take hours or even days, requiring significant computational resources. For many practitioners, this means constantly checking logs or monitoring progress to ensure scripts are running smoothly. But what happens when an error occurs hours into training, or when you need to know when a job has finished? Constant supervision is inefficient and distracting.

**KnockKnock**, a Python library by Hugging Face, solves this problem. By integrating with popular notification platforms like **Desktop Notifications**, **Telegram**, **Email**, and **Slack**, KnockKnock automatically informs you when your training starts, completes, or fails. This simple yet powerful tool streamlines your workflow, freeing you to focus on other tasks.

---

## The Problem and How KnockKnock Solves It

### The Challenge

Training machine learning models is resource-intensive and time-consuming. Practitioners often leave scripts running for hours or overnight, only to discover that:
- The script failed midway due to an error.
- Training completed hours ago, but valuable time was wasted waiting unnecessarily.
- Teams were not updated about training progress in real-time, leading to delays.

Manually monitoring scripts is a poor use of time, and it detracts from other critical activities.

### The Solution

KnockKnock simplifies the monitoring process by sending notifications directly to your preferred platform when:
- **Training Starts**: Know that your script is running successfully.
- **Training Ends**: Receive updates the moment your model completes training.
- **An Error Occurs**: Be alerted instantly to address issues promptly.

With just a few lines of code, KnockKnock ensures you stay informed wherever you are, saving you time and effort.

---

## Installing KnockKnock

Getting started with KnockKnock is easy. Install it using pip:

```bash
pip install knockknock
```

Ensure you’re using Python version 3.6 or higher for compatibility.

---

## Setting Up Notifications

KnockKnock supports a range of platforms to suit different needs. Here’s how to set up notifications for **Desktop Notifications**, **Telegram**, **Email**, and **Slack**.

### 1. Desktop Notifications

For simple on-screen updates, Desktop Notifications are an excellent choice. These are quick to set up and require no additional configuration.

#### Example Code:
```python
from knockknock import desktop_sender

@desktop_sender()
def train_model():
    print("Training started...")
    import time
    time.sleep(10)  # Simulate a long-running task
    print("Training complete!")
```

When the script runs, notifications will appear on your desktop for both the start and end of the training process.

---

### 2. Telegram Notifications

Telegram is perfect for receiving real-time updates on your phone or desktop.

#### Setup Instructions:
1. Create a bot on Telegram using **BotFather**.
2. Copy the API token provided by BotFather.
3. Retrieve your chat ID.

#### Example Code:
```python
from knockknock import telegram_sender

bot_token = "your_bot_token_here"
chat_id = "your_chat_id_here"

@telegram_sender(token=bot_token, chat_id=chat_id)
def train_model():
    print("Training started...")
    import time
    time.sleep(10)  # Simulate a long-running task
    print("Training complete!")
```

Once configured, you’ll receive a Telegram message whenever training starts, finishes, or fails.

---

### 3. Email Notifications

Stay updated through your inbox with email notifications.

#### Setup Instructions:
1. Install the `yagmail` library:
   ```bash
   pip install yagmail
   ```
2. Use your Gmail account to send notifications (it’s recommended to create a dedicated account for this purpose).

#### Example Code:
```python
from knockknock import email_sender

@email_sender(
    recipient_emails=["your_email@example.com"], 
    sender_email="your_gmail@gmail.com"
)
def train_model():
    print("Training started...")
    import time
    time.sleep(10)  # Simulate a long-running task
    print("Training complete!")
```

You’ll receive an email when your training starts and finishes, making it a reliable option for those who rely heavily on email updates.

---

### 4. Slack Notifications

For team collaborations, Slack notifications keep everyone informed in real-time.

#### Setup Instructions:
1. Create an **Incoming Webhook URL** for your Slack workspace.
2. Add the webhook URL to your code.

#### Example Code:
```python
from knockknock import slack_sender

webhook_url = "your_slack_webhook_url_here"

@slack_sender(webhook_url=webhook_url, channel="#your_channel")
def train_model():
    print("Training started...")
    import time
    time.sleep(10)  # Simulate a long-running task
    print("Training complete!")
```

Your Slack channel will receive updates whenever training begins, ends, or fails.

---

## Benefits of Using KnockKnock

1. **Save Time**: Eliminate the need to manually monitor training scripts, freeing you to focus on other tasks.
2. **Real-Time Alerts**: Get instant updates, even if you’re away from your workstation.
3. **Team Collaboration**: Keep your team in the loop with shared notifications via Slack or Telegram.
4. **Ease of Integration**: Add KnockKnock to your workflow with just a few lines of code.
5. **Customizable Platforms**: Choose from multiple notification methods to suit your personal or team preferences.

---

## Conclusion

KnockKnock is a game-changer for machine learning practitioners and teams. By automating training notifications, it eliminates the inefficiencies of manual monitoring, ensuring you’re always updated no matter where you are. Whether you prefer desktop notifications, Telegram updates, email alerts, or Slack messages, KnockKnock integrates seamlessly into your workflow.

Ready to make your workflow more efficient? [Check out KnockKnock on GitHub](https://github.com/huggingface/knockknock) and start automating your notifications today!

---

## References

- [KnockKnock GitHub Repository](https://github.com/huggingface/knockknock)
- [Slack Incoming Webhooks Guide](https://api.slack.com/messaging/webhooks)
- [Telegram BotFather Documentation](https://core.telegram.org/bots#botfather)
- [Yagmail Library](https://github.com/kootenpv/yagmail)
