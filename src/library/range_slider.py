import numpy as np
from matplotlib.widgets import AxesWidget


class RangeSlider(AxesWidget):
    """
    A range slider representing a floating point range.

    Create a slider from *valmin* to *valmax* in axes *ax*. For the slider to
    remain responsive you must maintain a reference to it. Call
    :meth:`on_changed` to connect to the slider event.

    Attributes
    ----------
    val : tupple of float
        Slider values.
    """

    def __init__(self, ax, label, valmin, valmax, valinit=(0.25, 0.75),
                 valfmt='%1.2f,%1.2f', closedmin=True, closedmax=True,
                 slidermin=None, slidermax=None, dragging=True, valstep=None,
                 orientation='horizontal', **kwargs):
        """
        Parameters
        ----------
        ax : Axes
            The Axes to put the slider in.

        label : str
            Slider label.

        valmin : float
            The minimum value of the slider.

        valmax : float
            The maximum value of the slider.

        valinit : float, optional, default: 0.5
            The slider initial position.

        valfmt : str, optional, default: "%1.2f,%1.2f"
            Used to format the slider value, fprint format string.

        closedmin : bool, optional, default: True
            Indicate whether the slider interval is closed on the bottom.

        closedmax : bool, optional, default: True
            Indicate whether the slider interval is closed on the top.

        slidermin : Slider, optional, default: None
            Do not allow the current slider to have a value less than
            the value of the Slider `slidermin`.

        slidermax : Slider, optional, default: None
            Do not allow the current slider to have a value greater than
            the value of the Slider `slidermax`.

        dragging : bool, optional, default: True
            If True the slider can be dragged by the mouse.

        valstep : float, optional, default: None
            If given, the slider will snap to multiples of `valstep`.

        orientation : str, 'horizontal' or 'vertical', default: 'horizontal'
            The orientation of the slider.

        Notes
        -----
        Additional kwargs are passed on to ``self.poly`` which is the
        :class:`~matplotlib.patches.Rectangle` that draws the slider
        knob.  See the :class:`~matplotlib.patches.Rectangle` documentation for
        valid property names (e.g., `facecolor`, `edgecolor`, `alpha`).
        """
        AxesWidget.__init__(self, ax)

        if slidermin is not None and not hasattr(slidermin, 'val'):
            raise ValueError("Argument slidermin ({}) has no 'val'"
                             .format(type(slidermin)))
        if slidermax is not None and not hasattr(slidermax, 'val'):
            raise ValueError("Argument slidermax ({}) has no 'val'"
                             .format(type(slidermax)))
        if orientation not in ['horizontal', 'vertical']:
            raise ValueError("Argument orientation ({}) must be either"
                             "'horizontal' or 'vertical'".format(orientation))

        self.orientation = orientation
        self.closedmin = closedmin
        self.closedmax = closedmax
        self.slidermin = slidermin
        self.slidermax = slidermax
        self.drag_active = False
        self.valmin = valmin
        self.valmax = valmax
        self.valstep = valstep
        val1 = self._value_in_bounds(valinit[0])
        val1 = valmin if val1 is None else val1
        val2 = self._value_in_bounds(valinit[1])
        val2 = valmax if val2 is None else val2
        valinit = (min(val1, val2), max(val1, val2))
        self.val = valinit
        self.valinit = valinit
        if orientation == 'vertical':
            self.poly = ax.axhspan(self.val[0], self.val[1], 0, 1, **kwargs)
            self.hline_l = ax.axhline(self.val[0], 0, 1, color='r', lw=1)
            self.hline_r = ax.axhline(self.val[1], 0, 1, color='r', lw=1)
        else:
            self.poly = ax.axvspan(self.val[0], self.val[1], 0, 1, **kwargs)
            self.vline_b = ax.axvline(self.val[0], 0, 1, color='r', lw=1)
            self.vline_t = ax.axvline(self.val[1], 0, 1, color='r', lw=1)
        self.valfmt = valfmt
        ax.set_yticks([])
        if orientation == 'vertical':
            ax.set_ylim((valmin, valmax))
        else:
            ax.set_xlim((valmin, valmax))
        ax.set_xticks([])
        ax.set_navigate(False)

        self.connect_event('button_press_event', self._update)
        self.connect_event('button_release_event', self._update)
        if dragging:
            self.connect_event('motion_notify_event', self._update)
        if orientation == 'vertical':
            self.label = ax.text(0.5, 1.02, label, transform=ax.transAxes,
                                 verticalalignment='bottom',
                                 horizontalalignment='center')

            self.valtext = ax.text(0.5, -0.02, valfmt % valinit,
                                   transform=ax.transAxes,
                                   verticalalignment='top',
                                   horizontalalignment='center')
        else:
            self.label = ax.text(-0.02, 0.5, label, transform=ax.transAxes,
                                 verticalalignment='center',
                                 horizontalalignment='right')

            self.valtext = ax.text(1.02, 0.5, valfmt % valinit,
                                   transform=ax.transAxes,
                                   verticalalignment='center',
                                   horizontalalignment='left')

        self.cnt = 0
        self.observers = {}

        # self.set_val(valinit)

    def _value_in_bounds(self, val):
        """Makes sure *val* is with given bounds."""
        if self.valstep:
            val = np.round((val - self.valmin) / self.valstep) * self.valstep
            val += self.valmin

        if val <= self.valmin:
            if not self.closedmin:
                return
            val = self.valmin
        elif val >= self.valmax:
            if not self.closedmax:
                return
            val = self.valmax

        if self.slidermin is not None and val <= self.slidermin.val:
            if not self.closedmin:
                return
            val = self.slidermin.val

        if self.slidermax is not None and val >= self.slidermax.val:
            if not self.closedmax:
                return
            val = self.slidermax.val
        return val

    def _update(self, event):
        """Update the slider position."""
        if self.ignore(event) or event.button != 1:
            return

        if event.name == 'button_press_event' and event.inaxes == self.ax:
            self.drag_active = True
            event.canvas.grab_mouse(self.ax)

        if not self.drag_active:
            return

        elif ((event.name == 'button_release_event') or (
              event.name == 'button_press_event' and event.inaxes != self.ax)):
            self.drag_active = False
            event.canvas.release_mouse(self.ax)
            return
        if self.orientation == 'vertical':
            val = self._value_in_bounds(event.ydata)
        else:
            val = self._value_in_bounds(event.xdata)

        valmid = (self.val[0] + self.val[1]) * 0.5
        if val < valmid:
            valnew = (val, self.val[1])
        else:
            valnew = (self.val[0], val)
        if val not in [None, *self.val]:
            self.set_val(valnew)

    def set_val(self, val):
        """
        Set slider value to *val*

        Parameters
        ----------
        val : float
        """
        self.val = val
        xy = self.poly.xy
        if self.orientation == 'vertical':
            xy[0] = 0, self.val[0]
            xy[1] = 0, self.val[1]
            xy[2] = 1, self.val[1]
            xy[3] = 1, self.val[0]
        else:
            xy[0] = self.val[0], 0
            xy[1] = self.val[0], 1
            xy[2] = self.val[1], 1
            xy[3] = self.val[1], 0
        self.poly.xy = xy
        self.valtext.set_text(self.valfmt % self.val)
        if self.drawon:
            self.ax.figure.canvas.draw_idle()
        if not self.eventson:
            return
        for cid, func in self.observers.items():
            func(self.val)

    def on_changed(self, func):
        """
        When the slider value is changed call *func* with the new
        slider value

        Parameters
        ----------
        func : callable
            Function to call when slider is changed.
            The function must accept a single float as its arguments.

        Returns
        -------
        cid : int
            Connection id (which can be used to disconnect *func*)
        """
        cid = self.cnt
        self.observers[cid] = func
        self.cnt += 1
        return cid

    def disconnect(self, cid):
        """
        Remove the observer with connection id *cid*

        Parameters
        ----------
        cid : int
            Connection id of the observer to be removed
        """
        try:
            del self.observers[cid]
        except KeyError:
            pass

    def reset(self):
        """Reset the slider to the initial value"""
        if self.val != self.valinit:
            self.set_val(self.valinit)
