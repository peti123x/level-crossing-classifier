
import smtplib, ssl, email, os
from email import encoders
from email.mime.base import MIMEBase
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
import time
import datetime

#Details
port = 587  # For starttls
smtp_server = "outlook.office365.com"
sender_email = "your email address"
receiver_email = "your email address"
password = input("Type your password and press enter:")
#password = "asd"
body = """This message is sent from Python."""

#Sends off the email
def sendMail(attach, tries=0, subject=None):
    ts = datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d-%H:%M:%S')
    #create attachment
    success = False
    print("Setting up ...")
    message = MIMEMultipart()
    message["From"] = sender_email
    message["To"] = receiver_email
    message["Subject"] = str(subject) + " " + ts
    message["Bcc"] = receiver_email
    #add body to email
    message.attach(MIMEText(body, "plain"))
    
    filename = attach
    #print("Open in binary ...")
    with open(filename, "rb") as attachment:
        part = MIMEBase("application", "octet-stream")
        part.set_payload(attachment.read())
    #print("Encoding ...")
    encoders.encode_base64(part)
    
    part.add_header(
        "Content-Disposition",
        f"attachment; filename= {filename}",
    )
    print("Adding attachment ...")
    message.attach(part)
    text = message.as_string()
    
    print("Logging in ...")
    context = ssl.create_default_context()
    with smtplib.SMTP(smtp_server, port) as server:
        print("Sending email ...")
        try:
            server.ehlo()  # Can be omitted
            server.starttls(context=context)
            server.ehlo()  # Can be omitted
            server.login(sender_email, password)
            server.sendmail(sender_email, receiver_email, text)
            server.quit()
        except Exception as e:
            print("Error occured:")
            print(e)
            if tries <= 2:
                sendMail(attach, tries+1)
                
        finally:
            print("Email sent.")
            success = True
    return success

#sendMail("trains.db")

