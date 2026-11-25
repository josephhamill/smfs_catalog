# Setting the InvOLS for many traces

**InvOLS** (inverse optical lever sensitivity, in nm/V) is the factor that
converts the detector signal into cantilever deflection. Every curve is
converted with one. By default each curve is converted with the value from its
own fit, made on the stretch of the approach where the tip is in contact with
the surface. The InvOLS window lets you instead decide one value for a group of
traces and apply it to all of them.

Open it from the dashboard with **InvOLS for these files...**. It works on the
files in the dashboard's current scope. Opening it, and closing it without
pressing Apply, changes nothing.

## The window

- **One point per trace.** Time is along the bottom, nm/V up the side. Each
  point is that trace's own-fit value, coloured by where the value in use comes
  from (own fit or ensemble). A trace with no own fit is not drawn. A point
  far from the rest is pinned to the edge as a triangle; hover to read its
  real value.
- **Three methods.** Pick one:
  - *Each trace's own fit*: use when contact is deep and clean on every trace,
    or the value may change between traces. Pressing Apply under this method
    reverts the window's traces to their own fits.
  - *Mean of all traces shown*: use when the scope has been narrowed to one
    cantilever and one laser alignment, and the points agree.
  - *Mean of the boxed traces*: use when some traces have a poor or missing
    contact region, or the setup changed partway. Click and drag a box on the
    plot; only the traces inside it count.
- **Average.** Choose which average is applied. The arithmetic mean is the
  default. The harmonic mean (the reciprocal of the mean of the reciprocals)
  differs from it by about the square of the relative scatter, and is the less
  biased of the two when the scatter comes from the per-curve slope fits. Both
  are shown; the one that will be applied is marked "applied". Your choice is
  remembered.
- **Readout.** The number of traces, both averages, the standard deviation of
  the per-trace values, and the standard error of the mean (the standard
  deviation divided by the square root of the number of traces), next to the
  instrument value stored in the files. The standard error covers scatter
  only. Errors shared by every trace, such as an error in the instrument value
  itself, are not in it.
- **Box lines.** If some traces inside the box's time span lie outside its
  nm/V limits, the readout says how many. The standard deviation and standard
  error then describe only the traces inside the limits. The app cannot tell
  whether those traces are failed fits or real scatter.
- **Apply.** Writes the value to every trace in the window, including traces
  with no own fit of their own. Traces whose file has no instrument value keep
  their own fit, and the window says how many. Revert with the first method.
- **Hollow points.** A point is hollow when its fit was made under earlier fit
  settings (the offset or window set in the window or the curve view). Hollow
  points are left out of the readout and of Apply until the queue visits the
  trace again.
- **Stale-settings warning.** Shown when an earlier apply used fit settings
  that differ from the ones now in force.
- **Open this trace.** Select a point, then open it in the curve view (a double
  click does the same). It adds the trace to the queue and pauses the worker.
  The curve view's label then reads, for example, `InvOLS in use 210.00 nm/V
  (ensemble of 3 traces)`.

## What applying does and does not do

Applying marks nothing and re-runs nothing. Forces and extensions change the
next time the queue visits each trace. Until then the dashboard's **Status**
column reads **stale params** for that trace, and the numbers shown for it are
still from the earlier value.

## Where you see it

- Dashboard column **InvOLS source**: `own fit`, or `ensemble of N traces,
  <date>`.
- Derived column **Transform InvOLS (nm/V)**: the value each trace is
  converted with.
- Queue-table export: `freshness` (up to date, stale params or not analysed,
  at the moment of export) and `invols_source`.

## 2D histograms leave these traces out

Both 2D histograms (physical and normalised) leave out every trace that has
an ensemble InvOLS, and say so in the window: "K left out: ensemble InvOLS".
They still convert each trace with its own InvOLS fit, so including such
traces would mix two conversions. A later change to share one conversion will
lift this. The normalised 2D histogram is a work in progress.

## What is kept, and what is not

Values that depend on a trace's ensemble neighbours, such as an average over
the traces now in the window, normally are not stored in the catalog; they live
only while a window is open. A decided InvOLS is the exception and is stored:
it belongs to the experiment's cantilever, which does not change, not to the
neighbours. Each apply also stores a group record (the traces, the average, the
numbers, the fit settings). Reverting keeps the records.

## One caution

If a file's spring constant came from a thermal tune made with that file's own
InvOLS, replacing the InvOLS leaves forces that are not self-consistent with
that spring constant. Per-curve slopes already do this today. The app does not
check for it.

## Upgrading an existing catalog

This feature moves the catalog from version 1 to version 2. On opening an
older catalog, the app first makes a backup, then runs every upgrade step as
one transaction, so a failure leaves the catalog as it was. Older builds
refuse a version 2 catalog.
