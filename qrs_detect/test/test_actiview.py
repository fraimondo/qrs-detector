import numpy                     # Used to calculate DFT
import matplotlib.pyplot as plt  # Used to plot DFT
import socket                    # used for TCP/IP communication
import struct

# TCP/IP setup
TCP_IP = '192.168.0.3' # ActiView is running on the same PC
TCP_PORT = 8888       # This is the port ActiView listens on
BUFFER_SIZE = 1968    # Data packet size (depends on ActiView settings)

# Open socket
s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
s.connect((TCP_IP, TCP_PORT))

# Create a 512-sample signal_buffer (arange fills the array with
# values, but these will be overwritten; We're just using arange
# to give us an array of the right size and type).
signal_buffer = numpy.arange(512)

# Calculate spectrum 50 times
for i in range(50):
    # Parse incoming frame data
    print("Parsing data")

    # Data buffer index (counts from 0 up to 512)
    buffer_idx = 0

    # collect 32 packets to fill the window
    # for n in range(32):
    # Read the next packet from the network
    data = s.recv(BUFFER_SIZE)

    # Extract 16 channel 1 samples from the packet
    for m in range(41):
        offset = m * 3 * 8
        # The 3 bytes of each sample arrive in reverse order
        sample = struct.unpack('<I', data[offset:offset+3] + b'\0')[0]
        # Store sample to signal buffer
        signal_buffer[buffer_idx] = sample
        print('{} -> {} ({})'.format(m, sample, (sample & 0x00FFFF00) / 256))
        buffer_idx += 1

    # print(signal_buffer[buffer_idx])

    # # Calculate DFT ("sp" stands for spectrum)
    # sp = numpy.fft.fft(signal_buffer)
    # sp[0] = 0 # eliminate DC component

    # Plot spectrum
    # print("Plotting data")
    # plt.plot(sp.real)
    # plt.hold(False)
    # plt.show()

# Close socket
s.close()
