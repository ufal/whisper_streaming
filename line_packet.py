#!/usr/bin/env python3

"""Functions for sending and receiving individual lines of text over a socket.

A line is transmitted using one or more fixed-size packets of UTF-8 bytes
containing:

  - Zero or more bytes of UTF-8, excluding \n and \0, followed by

  - Zero or more \0 bytes as required to pad the packet to PACKET_SIZE

Originally from the UEDIN team of the ELITR project. 
"""

PACKET_SIZE = 65536


def send_one_line(socket, text, pad_zeros=False):
    """Sends a line of text over the given socket.

    The 'text' argument should contain a single line of text (line break
    characters are optional). Line boundaries are determined by Python's
    str.splitlines() function [1]. We also count '\0' as a line terminator.
    If 'text' contains multiple lines then only the first will be sent.

    If the send fails then an exception will be raised.

    [1] https://docs.python.org/3.5/library/stdtypes.html#str.splitlines

    Args:
        socket: a socket object.
        text: string containing a line of text for transmission.
    """
    text.replace('\0', '\n')
    lines = text.splitlines()
    first_line = '' if len(lines) == 0 else lines[0]
    # TODO Is there a better way of handling bad input than 'replace'?
    data = first_line.encode('utf-8', errors='replace') + b'\n' + (b'\0' if pad_zeros else b'')
    for offset in range(0, len(data), PACKET_SIZE):
        bytes_remaining = len(data) - offset
        if bytes_remaining < PACKET_SIZE:
            padding_length = PACKET_SIZE - bytes_remaining
            packet = data[offset:] + (b'\0' * padding_length if pad_zeros else b'')
        else:
            packet = data[offset:offset+PACKET_SIZE]
        socket.sendall(packet)


def receive_one_line(socket):
    """Receives a line of text from the given socket.

    This function will (attempt to) receive a single line of text. If data is
    currently unavailable then it will block until data becomes available or
    the sender has closed the connection (in which case it will return an
    empty string).

    The string should not contain any newline characters, but if it does then
    only the first line will be returned.

    Args:
        socket: a socket object.

    Returns:
        A string representing a single line with a terminating newline or
        None if the connection has been closed.
    """
    data = b''
    while True:
        packet = socket.recv(PACKET_SIZE)
        if not packet:  # Connection has been closed.
            return None
        data += packet
        if b'\0' in packet:
            break
    # TODO Is there a better way of handling bad input than 'replace'?
    text = data.decode('utf-8', errors='replace').strip('\0')
    lines = text.split('\n')
    return lines[0] + '\n'


def receive_lines(socket):
    try:
        data = socket.recv(PACKET_SIZE)
    except BlockingIOError:
        return []
    if data is None:  # Connection has been closed.
        return None
    # TODO Is there a better way of handling bad input than 'replace'?
    text = data.decode('utf-8', errors='replace').strip('\0')
    lines = text.split('\n')
    if len(lines)==1 and not lines[0]:
        return None
    return lines
