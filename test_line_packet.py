import line_packet


class Sock:
    def __init__(self):
        self.buf = bytearray()

    def sendall(self, packet):
        self.buf += packet


def sent_line(text):
    sock = Sock()
    line_packet.send_one_line(sock, text)
    return bytes(sock.buf).split(b"\n", 1)[0]


def test_null_ends_the_line():
    assert sent_line("hello\0world") == b"hello"


def test_newline_still_ends_the_line():
    assert sent_line("hello\nworld") == b"hello"
