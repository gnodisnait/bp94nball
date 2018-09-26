
import numpy as np
import matplotlib.pyplot as plt


def cos_of_vec(u,w):
    return np.dot(u,w)/vec_length(u)/vec_length(w)


def vec_length(v):
    return np.sqrt(np.dot(v,v))


def dis_between(v1, v2):
    return np.sqrt(np.dot(v1 - v2, v1 - v2))


def dis_between_ball_centers(ball1, ball2):
    v1 = np.multiply(ball1[:2], ball1[-2])
    v2 = np.multiply(ball2[:2], ball2[-2])
    return dis_between(v1, v2)


def get_qsr(ball1, ball2):
    """
    compute the relation between ball1 and ball2
    :param ball1:
    :param ball2:
    :return: ['part_of']
    """
    r1 = ball1[-1]
    r2 = ball2[-1]
    d = dis_between_ball_centers(ball1, ball2)
    if r1 + d <= r2:
        return 'part_of'
    if r1 + r2 < d:
        return 'disconnect'
    return 'unknown'


def qsr_part_of_characteristic_function(ball1, ball2):
    """
    ball1, ball2 are vectors in the form of  alpha1, l1, r1 = ball1[:-2], ball1[-2], ball1[-1]
    alpha1 is a unit point on the unit ball!
    distance between the center points of ball1 and ball2 + radius of ball1 - radius of ball2
    return <=0 ball1 part of ball2
           > 0 ball1 not part of ball2

    :param ball1:
    :param ball2:
    :return: R
    """
    alpha1, l1, r1 = ball1[:-2], ball1[-2], ball1[-1]
    alpha2, l2, r2 = ball2[:-2], ball2[-2], ball2[-1]
    return dis_between(np.multiply(l1, alpha1), np.multiply(l2, alpha2)) + r1 - r2

def qsr_disconnect_characteristic_function(ball1, ball2):
    """
    ball1, ball2 are vectors in the form of  alpha1, l1, r1 = ball1[:-2], ball1[-2], ball1[-1]
    alpha1 is a unit point on the unit ball!
    distance between the center points of ball1 and ball2 + radius of ball1 - radius of ball2
    return <=0 ball1 disconects from ball2
           > 0 ball1 does not disconnect from ball2

    :param ball1:
    :param ball2:
    :return: R
    """
    alpha1, l1, r1 = ball1[:-2], ball1[-2], ball1[-1]
    alpha2, l2, r2 = ball2[:-2], ball2[-2], ball2[-1]
    return r1 + r2 - dis_between(np.multiply(l1, alpha1), np.multiply(l2, alpha2))


def circles(x, y, s, c='b', vmin=None, vmax=None, **kwargs):
    """
    Make a scatter of circles plot of x vs y, where x and y are sequence
    like objects of the same lengths. The size of circles are in data scale.

    Parameters
    ----------
    x,y : scalar or array_like, shape (n, )
        Input data
    s : scalar or array_like, shape (n, )
        Radius of circle in data unit.
    c : color or sequence of color, optional, default : 'b'
        `c` can be a single color format string, or a sequence of color
        specifications of length `N`, or a sequence of `N` numbers to be
        mapped to colors using the `cmap` and `norm` specified via kwargs.
        Note that `c` should not be a single numeric RGB or RGBA sequence
        because that is indistinguishable from an array of values
        to be colormapped. (If you insist, use `color` instead.)
        `c` can be a 2-D array in which the rows are RGB or RGBA, however.
    vmin, vmax : scalar, optional, default: None
        `vmin` and `vmax` are used in conjunction with `norm` to normalize
        luminance data.  If either are `None`, the min and max of the
        color array is used.
    kwargs : `~matplotlib.collections.Collection` properties
        Eg. alpha, edgecolor(ec), facecolor(fc), linewidth(lw), linestyle(ls),
        norm, cmap, transform, etc.

    Returns
    -------
    paths : `~matplotlib.collections.PathCollection`

    Examples
    --------
    a = np.arange(11)
    circles(a, a, a*0.2, c=a, alpha=0.5, edgecolor='none')
    plt.colorbar()

    License
    --------
    This code is under [The BSD 3-Clause License]
    (http://opensource.org/licenses/BSD-3-Clause)
    """
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle
    from matplotlib.collections import PatchCollection
    c0 = c
    if np.isscalar(c):
        kwargs.setdefault('color', c)
        c = None
    if 'fc' in kwargs: kwargs.setdefault('facecolor', kwargs.pop('fc'))
    if 'ec' in kwargs: kwargs.setdefault('edgecolor', kwargs.pop('ec'))
    if 'ls' in kwargs: kwargs.setdefault('linestyle', kwargs.pop('ls'))
    if 'lw' in kwargs: kwargs.setdefault('linewidth', kwargs.pop('lw'))

    patches = [Circle((x_, y_), s_) for x_, y_, s_ in np.broadcast(x, y, s)]
    collection = PatchCollection(patches, **kwargs)
    if c is not None:
        collection.set_array(np.asarray(c))
        collection.set_clim(vmin, vmax)

    ax = plt.gca()
    ax.add_collection(collection)
    if 'label' in kwargs.keys():
        ax.text(x,y, kwargs['label'], color = c0 )

    ax.autoscale_view()
    if c is not None:
        plt.sci(collection)
    return collection


class DynamicUpdate():
    #Suppose we know the x range
    min_x = 0
    max_x = 10

    def on_launch(self):
        #Set up plot
        self.figure, self.ax = plt.subplots()
        self.lines, = self.ax.plot([],[], 'o')
        #Autoscale on unknown axis and known lims on the other
        self.ax.set_autoscaley_on(True)
        self.ax.set_xlim(self.min_x, self.max_x)
        #Other stuff
        self.ax.grid()

    def on_running(self, xdata, ydata):
        #Update data (with the new _and_ the old points)
        self.lines.set_xdata(xdata)
        self.lines.set_ydata(ydata)
        #Need both of these in order to rescale
        self.ax.relim()
        self.ax.autoscale_view()
        #We need to draw *and* flush
        self.figure.canvas.draw()
        self.figure.canvas.flush_events()

    #Example
    def __call__(self):
        import numpy as np
        import time
        self.on_launch()
        xdata = []
        ydata = []
        for x in np.arange(0,10,0.5):
            xdata.append(x)
            ydata.append(np.exp(-x**2)+10*np.exp(-(x-7)**2))
            self.on_running(xdata, ydata)
            time.sleep(1)
        return xdata, ydata
