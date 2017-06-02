import numpy as np

from vispy import app, gloo, keys


import sys
sys.path.append('../../../../fieldtrip/realtime/src/buffer/python')
sys.path.append('../../fieldtrip/realtime/src/buffer/python')
sys.path.append('/Users/fraimondo/dev/fieldtrip/realtime/src/buffer/python')
import FieldTrip as ft

from argparse import ArgumentParser

parser = ArgumentParser(description='Visualize QRS Exp from fieldtrip buffer')
parser.add_argument('--host', metavar='host', type=str, nargs=1,
                    default=['localhost'],
                    help='fieldtrip buffer host (default localhost)')
parser.add_argument('--port', metavar='port', type=int, nargs=1,
                    default=[1972],
                    help='fieldtrip buffer port (default 1972)')

parser.add_argument('--scale', metavar='scale', type=int, nargs=1,
                    default=[8000],
                    help='fieldtrip buffer port (default 1972)')

parser.add_argument('--filter', dest='filter', action='store_true',
                    default=False,
                    help='Filter original data')

args = parser.parse_args()


host = args.host[0]
port = args.port[0]

if args.filter is True:
    sys.path.append('./iir1/')
    import pyiir1 as iir


class RingBuffer(object):
    def __init__(self, max_size, rows=1, dtype=np.float):
        self.size = 0
        self.max_size = max_size
        self.buffer = np.zeros((self.max_size, rows), dtype=dtype)
        self.counter = 0

    def append(self, data, auto=False):
        """this is an O(n) operation"""
        n = len(data)
        # if auto is True:
        #   print self.remaining, self.size, '->',
        if self.max_size - len(self) < n:
            if auto is True:
                # print 'Auto consume', n - (self.max_size - len(self)), '->',
                self.consume(n - (self.max_size - len(self)))
            else:
                raise RuntimeError("Buffer Overflow")
        if self.remaining < n:
            # print 'compacting',
            self.compact()
        # if auto is True:
            # print self.remaining, self.size
        if isinstance(data, np.ndarray):
            if len(data.shape) < 2:
                self.buffer[self.counter + self.size:][:n] = data[:, None]
            else:
                self.buffer[self.counter + self.size:][:n] = data
        else:
            self.buffer[self.counter + self.size:][:n] = \
                np.array(data)[:, None]

        self.size += n

    def consume(self, n):
        self.counter += n
        self.size -= n

    @property
    def remaining(self):
        return self.max_size - (self.counter + self.size)

    def compact(self):
        """
        note: only when this function is called, is an O(size)
        performance hit incurred,
        and this cost is amortized over the whole padding space
        """
        self.buffer[:self.size] = self.view
        self.counter = 0

    def ravel(self):
        return self.view.ravel()

    @property
    def view(self):
        """this is always an O(1) operation"""
        return self.buffer[self.counter:][:self.size]

    def cumsum(self, axis, dtype, out):
        return np.cumsum(self.buffer[self.counter:][:self.size],
                         axis, dtype, out)

    def mean(self, axis, dtype, out, keepdims=False):
        return np.mean(self.buffer[self.counter:][:self.size],
                       axis, dtype, out, keepdims)

    # def argmax(self, axis):
    #     return np.argmax(self.buffer[self.counter:][:self.size], axis)

    def __len__(self):
        return self.size

    def __getitem__(self, index):
        # import pdb; pdb.set_trace()
        if isinstance(index, slice):
            start = index.start
            stop = index.stop
        elif isinstance(index, int):
            start = index
            stop = index + 1
        if start is None:
            start = 0
        if stop is None:
            stop = 0
        if start < 0:
            start += self.size
            stop += self.size
        if stop > self.size:
            raise RuntimeError("Slice end cannot be bigger than size")
        elif stop < 0:
            stop += self.size

        if start < 0:
            raise RuntimeError("Out of bounds")
        stop += self.counter
        start += self.counter
        return self.buffer[start:stop]

    def __repr__(self):
        return self.view.__repr__()

print('Reading data from {}:{}'.format(host, port))
ftc = ft.Client()
ftc.connect(host, port)

H = ftc.getHeader()
if H is None:
    print('Failed to retrieve header!')
    sys.exit(1)

print(H)
print(H.labels)

# Config
sfreq = H.fSample
buffer_size = 4
MAX_SAMPS = 100000

x_scale_init = int(5 * sfreq)  # Number of samples to display
y_scale_init = np.float(args.scale[0])  # Amplitude
y_offset_init = 0  # -8190.0 # -200.0  # Offset

# End config

DELAYS = {}
DELAYS[250.0] = 12
DELAYS[500.0] = 4
DELAYS[1000.0] = 8
DELAYS[5000.0] = 389

SIGNAL_VERT_SHADER = """
#version 120
attribute float signal_pos;
attribute float index;
uniform int n_rows;
uniform int row;
uniform float width;
uniform float height;
uniform float y_scale;
uniform float y_offset;
void main (void)
{
    float offset = 1.0 - (2.0 / n_rows * (row + 0.5));
    gl_Position = vec4(index * 1.5 + 0.5,
                       ((signal_pos + y_offset) / y_scale) / n_rows + offset,
                       0.0, 1.0);
}
"""

SIGNAL_FRAG_SHADER = """
#version 120
uniform vec4 COLOR_MASKS[ 8 ] = vec4[] (vec4( 0.0, 0.0, 0.0, 1.0 ),
                                vec4( 1.0, 0.0, 0.0, 1.0 ),
                                vec4( 0.0, 1.0, 0.0, 1.0 ),
                                vec4( 0.0, 0.0, 1.0, 1.0 ),
                                vec4( 1.0, 1.0, 0.0, 1.0 ),
                                vec4( 1.0, 0.0, 1.0, 1.0 ),
                                vec4( 0.0, 1.0, 1.0, 1.0 ),
                                vec4( 1.0, 1.0, 1.0, 1.0 ) );
uniform int color;
void main()
{
    gl_FragColor = COLOR_MASKS[color];
}
"""

MARKER_VERT_SHADER = """
#version 120
attribute float markers;
attribute float y_pos;
uniform float start_samp;
uniform float width;
uniform float height;
uniform float x_scale;
void main (void)
{

    gl_Position = vec4((markers - start_samp)/x_scale * 1.5 + 0.5, y_pos, 0.0, 1.0);
}
"""

MARKER_FRAG_SHADER = """
#version 120
uniform vec4 COLOR_MASKS[ 8 ] = vec4[] (vec4( 0.0, 0.0, 0.0, 1.0 ),
                                vec4( 1.0, 0.0, 0.0, 1.0 ),
                                vec4( 0.0, 1.0, 0.0, 1.0 ),
                                vec4( 0.0, 0.0, 1.0, 1.0 ),
                                vec4( 1.0, 1.0, 0.0, 1.0 ),
                                vec4( 1.0, 0.0, 1.0, 1.0 ),
                                vec4( 0.0, 1.0, 1.0, 1.0 ),
                                vec4( 1.0, 1.0, 1.0, 1.0 ) );
uniform int color;
void main()
{
    gl_FragColor = COLOR_MASKS[color];
}
"""


class Canvas(app.Canvas):
    def __init__(self, do_filter=False):
        app.Canvas.__init__(self, title='Realtime ECG Detection',
                            keys='interactive')
        self.signal_program = gloo.Program(SIGNAL_VERT_SHADER,
                                           SIGNAL_FRAG_SHADER)
        self.marker_program = gloo.Program(MARKER_VERT_SHADER,
                                           MARKER_FRAG_SHADER)

        self.samps = RingBuffer(MAX_SAMPS, rows=3, dtype=np.float32)
        self.samps.append(np.zeros((x_scale_init + 1, 3)))

        self.x_scale = x_scale_init
        self.y_scale = y_scale_init
        self.y_offset = y_offset_init

        # self.signal_program['index'] = (
        #     np.arange(-x_scale_init, 0, 1) / x_scale_init).astype(np.float32)
        # self.signal_program['signal_pos'] = self.samps[-x_scale_init:]
        self.signal_program['y_scale'] = self.y_scale
        self.signal_program['y_offset'] = self.y_offset
        self.signal_program['n_rows'] = 3

        self.p_peaks = np.array([]).astype(np.float32)
        self.d_peaks = np.array([]).astype(np.float32)
        self.e_peaks = np.array([]).astype(np.float32)
        self.m_peaks = np.array([]).astype(np.float32)
        self._timer = app.Timer(1 / sfreq, connect=self.on_timer,
                                start=True)
        self.prev_sample = 0
        self.this_samp = 0
        self.do_filter = do_filter
        self.filter_delay = 0
        if do_filter:
            self.iir = None

    def reset(self):
        print('Resetting data')
        self.samps = RingBuffer(MAX_SAMPS, rows=3, dtype=np.float32)
        self.samps.append(np.zeros((x_scale_init + 1, 3)))
        self.p_peaks = np.array([]).astype(np.float32)
        self.d_peaks = np.array([]).astype(np.float32)
        self.e_peaks = np.array([]).astype(np.float32)
        self.m_peaks = np.array([]).astype(np.float32)
        self.prev_sample = 0
        self.this_samp = 0

    def on_initialize(self, event):
        gloo.set_state(clear_color='black', blend=True,
                       blend_func=('src_alpha', 'one_minus_src_alpha'))

    def on_resize(self, event):
        self.width, self.height = event.size
        self.signal_program['height'] = self.height
        self.signal_program['width'] = self.width
        self.marker_program['height'] = self.height
        self.marker_program['width'] = self.width
        print('H: {} - W: {}'.format(self.height, self.width))
        gloo.set_viewport(0, 0, self.width, self.height)

    def on_timer(self, event):
        newH = ftc.getHeader()
        if self.do_filter and self.iir is None:
            # Filter delay was computed for 0.5, 45.0 butterworth bandpass
            # at 250 Hz.
            self.filter_delay = DELAYS[newH.fSample]
            self.iir = [iir.ButterworthBandPass(3, newH.fSample, 8, 20.0)
                        for _ in range(3)]
            for i in range(3):
                self.iir[i].reset()
        new_samples = newH.nSamples - self.prev_sample
        if (new_samples > 0):
            beg = (self.prev_sample if new_samples < MAX_SAMPS
                   else newH.nSamples - MAX_SAMPS)

            data = ftc.getData([beg, newH.nSamples - 1])
            samps = np.copy(data[:, 0:3])
            if self.do_filter:
                for i in range(0, 1):
                    this_samps = samps[:, i].tolist()
                    self.iir[i].filter(this_samps)
                    samps[:, i] = np.array(this_samps)
            self.samps.append(samps, auto=True)
            self.this_samp = data[:, 3][-1]
            new_pp = data[:, 4]
            new_pp = new_pp[new_pp != 0]
            if len(new_pp) > 0:
                for p in new_pp:
                    cp = np.float32(p + self.filter_delay)
                    if cp not in self.p_peaks:
                        self.p_peaks = np.append(self.p_peaks, cp)
            new_dp = data[:, 5]
            new_dp = new_dp[new_dp != 0]
            if len(new_dp) > 0:
                self.d_peaks = np.append(self.d_peaks,
                                         new_dp + self.filter_delay)
            new_mp = data[:, 6]
            new_mp = new_mp[new_mp != 0]
            if len(new_mp) > 0:
                self.m_peaks = np.append(self.m_peaks,
                                         new_mp + self.filter_delay)

            self.prev_sample = newH.nSamples

        self.p_peaks = self.p_peaks[
            self.p_peaks > (self.this_samp - self.x_scale)]
        self.d_peaks = self.d_peaks[
            self.d_peaks > (self.this_samp - self.x_scale)]
        self.e_peaks = self.e_peaks[
            self.e_peaks > (self.this_samp - self.x_scale)]
        if self.p_peaks.shape[0] > 0 and self.d_peaks.shape[0] > 0:
            equal = np.intersect1d(self.p_peaks, self.d_peaks)
            if len(equal) > 0:
                self.e_peaks = np.append(self.e_peaks, equal)
                self.d_peaks = np.delete(self.d_peaks, equal)
                self.p_peaks = np.delete(self.p_peaks, equal)

        self.marker_program['start_samp'] = self.this_samp
        self.marker_program['x_scale'] = self.x_scale

        self.update()

    def on_draw(self, event):
        gloo.clear()
        index = (
            np.arange(-self.x_scale, 0, 1) / self.x_scale).astype(np.float32)
        index_shifted = (
            np.arange(-self.x_scale + self.filter_delay, self.filter_delay, 1) /
            self.x_scale).astype(np.float32)
        for i, index, color in zip(
                range(3), (index, index_shifted, index_shifted), [2, 7, 6]):
            self.signal_program['index'] = index
            self.signal_program['row'] = i
            self.signal_program['color'] = color
            self.signal_program['signal_pos'] = \
                self.samps[-self.x_scale:][:, i].ravel()
            self.signal_program.draw('line_strip')
        if self.p_peaks.shape[0] > 0:
            self.marker_program['markers'] = np.repeat(self.p_peaks, 2)
            self.marker_program['y_pos'] = np.tile(
                [-1, 1], self.p_peaks.shape[0]).astype(np.float32)
            self.marker_program['color'] = 1
            self.marker_program.draw('lines')
        if self.d_peaks.shape[0] > 0:
            self.marker_program['markers'] = np.repeat(self.d_peaks, 2)
            self.marker_program['y_pos'] = np.tile(
                [-1, 1], self.d_peaks.shape[0]).astype(np.float32)
            self.marker_program['color'] = 3
            self.marker_program.draw('lines')
        if self.e_peaks.shape[0] > 0:
            self.marker_program['markers'] = np.repeat(self.e_peaks, 2)
            self.marker_program['y_pos'] = np.tile(
                [-1, 1], self.e_peaks.shape[0]).astype(np.float32)
            self.marker_program['color'] = 4
            self.marker_program.draw('lines')
        if self.m_peaks.shape[0] > 0:
            self.marker_program['markers'] = np.repeat(self.m_peaks, 2)
            self.marker_program['y_pos'] = np.tile(
                [-1, 1], self.m_peaks.shape[0]).astype(np.float32)
            self.marker_program['color'] = 6
            self.marker_program.draw('lines')

    def on_mouse_wheel(self, event):
        # self.print_mouse_event(event, 'Mouse wheel')
        delta = int(10 * event.delta[1])
        self.y_scale = max(1, self.y_scale + delta)
        self.signal_program['y_scale'] = self.y_scale
        print('Scale: {}'.format(self.y_scale))

    def on_mouse_press(self, event):
        if event.button == 1:
            self.last_pos = event.pos[1]
        # self.print_mouse_event(event, 'Mouse press')

    def on_mouse_release(self, event):
        if event.button == 1 and self.last_pos is not None:
            delta = event.pos[1] - self.last_pos
            self.y_offset -= 10 * delta
            self.signal_program['y_offset'] = self.y_offset
            print('Offset: {}'.format(self.y_offset))
        self.last_pos = None
        # self.print_mouse_event(event, 'Mouse release')

    def print_mouse_event(self, event, what):
        modifiers = ', '.join([key.name for key in event.modifiers])
        print('{} - pos: {}, button: {}, modifiers: {}, delta: {}'.format(
              what, event.pos, event.button, modifiers, event.delta))

    def change_time_scale(self, value):
        self.x_scale = max(2, self.x_scale + value)

    def on_key_press(self, event):
        if event.key is keys.SPACE:
            if self._timer.running:
                self._timer.stop()
            else:
                self._timer.start()
        elif event.key == keys.LEFT:
            self.change_time_scale(-1)
        elif event.key == keys.RIGHT:
            self.change_time_scale(1)
        elif event.key == 'r':
            self.reset()
        else:
            modifiers = [key.name for key in event.modifiers]
            print('Key pressed - text: %r, key: %s, modifiers: %r' % (
                  event.text, event.key.name
                  if event.key is not None else '', modifiers))

canvas = Canvas(do_filter=args.filter)
canvas.show()
app.run()
