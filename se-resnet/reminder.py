import smtplib
from email.mime.text import MIMEText
from email.header import Header


class Reminder:
    def __init__(self, qq=None, register=None):
        """
        :param qq: 发送的qq账号
        :param register: qq邮箱授权吧
        """
        self.qq = qq
        self.register = register
        self.server = smtplib.SMTP_SSL("smtp.qq.com", 465)

    def send(self, title=None, detail=None):
        """
        send message
        :param title: the title of the message
        :param detail: the detail of the message
        """
        sender = self.qq
        receivers = self.qq
        message = MIMEText(detail, 'plain', 'utf-8')
        message['Subject'] = Header(title, 'utf-8')
        message['From'] = sender
        message['To'] = receivers
        try:
            self.server = smtplib.SMTP_SSL("smtp.qq.com", 465)
            self.server.login(sender, self.register)
            self.server.sendmail(sender, receivers, message.as_string())
            self.server.quit()
        except smtplib.SMTPException as e:
            print(e)


    def _register(self):
        self.qq = '434596665@qq.com'
        self.register = 'qbcomikcojwubgca'

if __name__ == '__main__':

    reminder = Reminder()
    reminder._register()
    title = 'info test'
    subject = '测试成功\naaa'
    reminder.send(title,subject)

