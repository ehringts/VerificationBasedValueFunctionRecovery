"""Observer utilities for storing RKHS-PI errors and plotting the comparison between standard and product kernels."""
import copy
import pickle
import numpy as np

class Observer:
    """Store training, value, performance, and quadratic-bound errors and create comparison plots."""

    def __init__(self, name):
        """Initialize the object and precompute the quantities used later."""
        self.trueErrorListStandart     = []
        self.performanceListStandart   = []
        self.resErrorListStandart      = []
        self.stagListStandart          = []
        self.GreedyErrorListStandart   = []
        self.trueErrorListQuadKernel   = []
        self.performanceListQuadKernel = []
        self.resErrorListQuadKernel    = []
        self.stagListQuadKernel        = []
        self.GreedyErrorListQuadKernel = []
        self.upperListQuadKernel       = []
        self.lowerListQuadKernel       = []
        self.upperListStandart         = []
        self.lowerListStandart         = []
        self.name                      = name
        self.perfErrorListStandart     = []
        self.perfErrorListQuadKernel   = []

    def saveObserver(self, name):
        """Save the observer object to disk."""
        with open('data/' + name, 'wb') as outp:
            pickle.dump(self, outp, protocol=4)

    def loadObserver(self, name):
        """Load observer data from disk."""
        with open('data/' + name, 'rb') as inp:
            oldSelf                        = pickle.load(inp)
            self.trueErrorListStandart     = oldSelf.trueErrorListStandart
            self.performanceListStandart   = oldSelf.performanceListStandart
            self.resErrorListStandart      = oldSelf.resErrorListStandart
            self.GreedyErrorListStandart   = oldSelf.GreedyErrorListStandart
            self.stagListStandart          = oldSelf.stagListStandart
            self.trueErrorListQuadKernel   = oldSelf.trueErrorListQuadKernel
            self.performanceListQuadKernel = oldSelf.performanceListQuadKernel
            self.resErrorListQuadKernel    = oldSelf.resErrorListQuadKernel
            self.GreedyErrorListQuadKernel = oldSelf.GreedyErrorListQuadKernel
            self.stagListQuadKernel        = oldSelf.stagListQuadKernel
            self.upperListQuadKernel       = oldSelf.upperListQuadKernel
            self.lowerListQuadKernel       = oldSelf.lowerListQuadKernel
            self.upperListStandart         = oldSelf.upperListStandart
            self.lowerListStandart         = oldSelf.lowerListStandart
            self.perfErrorListStandart     = oldSelf.perfErrorListStandart
            self.perfErrorListQuadKernel   = oldSelf.perfErrorListQuadKernel
            self.name                      = oldSelf.name

    def addObjectGreedyErrorStandart(self, GreedyError):
        """Store the latest standard-kernel greedy residual."""
        self.GreedyErrorListStandart.append(copy.copy(GreedyError))
        if self.name:
            self.saveObserver(self.name)

    def addObjectGreedyErrorQuadKernel(self, GreedyError):
        """Store the latest product-kernel greedy residual."""
        self.GreedyErrorListQuadKernel.append(copy.copy(GreedyError))
        if self.name:
            self.saveObserver(self.name)

    def addObjectStandart(self, performance, resError, stag):
        """Store one standard-kernel policy-iteration error triple."""
        self.performanceListStandart.append(copy.copy(performance))
        self.resErrorListStandart.append(copy.copy(resError))
        self.stagListStandart.append(copy.copy(stag))
        if self.name:
            self.saveObserver(self.name)

    def addObjectQuadKernel(self, performance, resError, stag):
        """Store one product-kernel policy-iteration error triple."""
        self.performanceListQuadKernel.append(copy.copy(performance))
        self.resErrorListQuadKernel.append(copy.copy(resError))
        self.stagListQuadKernel.append(copy.copy(stag))
        if self.name:
            self.saveObserver(self.name)

    def addPerfomErrorQuadKernel(self, comSum):
        """Store one product-kernel performance error."""
        self.perfErrorListQuadKernel.append(copy.copy(comSum))
        if self.name:
            self.saveObserver(self.name)

    def addPerfomErrorStandart(self, comSum):
        """Store one standard-kernel performance error."""
        self.perfErrorListStandart.append(copy.copy(comSum))
        if self.name:
            self.saveObserver(self.name)

    def addTrueErrorStandart(self, error):
        """Store one standard-kernel value-function error."""
        self.trueErrorListStandart.append(copy.copy(error))
        if self.name:
            self.saveObserver(self.name)

    def addTrueErrorQuadKernel(self, error):
        """Store one product-kernel value-function error."""
        self.trueErrorListQuadKernel.append(copy.copy(error))
        if self.name:
            self.saveObserver(self.name)

    def addQuadraticStandart(self, lower, upper):
        """Store one standard-kernel quadratic lower and upper bound."""
        self.lowerListStandart.append(copy.copy(lower))
        self.upperListStandart.append(copy.copy(upper))
        if self.name:
            self.saveObserver(self.name)

    def addQuadraticQuadKernel(self, lower, upper):
        """Store one product-kernel quadratic lower and upper bound."""
        self.lowerListQuadKernel.append(copy.copy(lower))
        self.upperListQuadKernel.append(copy.copy(upper))
        if self.name:
            self.saveObserver(self.name)

    def plotObserver(self, filename, plot_title, lower, upper):
        """Plot greedy, value, performance, and quadratic-bound errors for the low-dimensional examples."""
        import numpy as np
        import matplotlib.pyplot as plt
        from matplotlib.ticker import MaxNLocator, FuncFormatter, LogFormatterMathtext, NullFormatter, NullLocator
        rc = {'text.usetex': True, 'font.family': 'serif', 'font.serif': ['Computer Modern Roman', 'CMU Serif', 'Latin Modern Roman', 'DejaVu Serif'], 'axes.titlesize': 15, 'axes.labelsize': 13, 'xtick.labelsize': 12, 'ytick.labelsize': 12, 'legend.fontsize': 15, 'lines.linewidth': 1.6, 'text.latex.preamble': '\\usepackage{amsmath,amssymb}'}
        with plt.rc_context(rc):
            wid        = 1.6
            color_std  = '#ff7f0e'
            color_quad = 'C2'
            color_ref  = 'black'
            fig        = plt.figure(figsize=(12, 6.8))
            fig.suptitle(plot_title, fontsize=20, y=0.955)
            left_margin       = 0.065
            right_margin      = 0.02
            column_gap        = 0.075
            column_width      = (1.0 - left_margin - right_margin - column_gap) / 2.0
            left_x            = left_margin
            right_x           = left_margin + column_width + column_gap
            title_pad         = 10
            xlabel_pad        = 6
            ylabel_pad        = 5
            vertical_gap_inch = 0.9
            vertical_gap      = vertical_gap_inch / fig.get_figheight()
            bottom_row_bottom = 0.09
            bottom_row_height = 0.13
            bottom_row_top    = bottom_row_bottom + bottom_row_height
            upper_bottom      = bottom_row_top + vertical_gap
            upper_top         = 0.84
            upper_height      = upper_top - upper_bottom
            ax1               = fig.add_axes([left_x, upper_bottom, column_width, upper_height])
            pos1              = ax1.get_position()
            right_gap         = vertical_gap
            right_plot_height = (pos1.height - right_gap) / 2.0
            ax2               = fig.add_axes([right_x, pos1.y1 - right_plot_height, column_width, right_plot_height])
            ax5               = fig.add_axes([right_x, pos1.y0, column_width, right_plot_height])
            ax3               = fig.add_axes([left_x, bottom_row_bottom, column_width, bottom_row_height])
            ax4               = fig.add_axes([right_x, bottom_row_bottom, column_width, bottom_row_height])

            def set_zoomed_yaxis(ax, *arrays, pad_ratio=0.15, decimal=False):
                """Set a zoomed linear y-axis for finite plotting data."""
                y_all = []
                for arr in arrays:
                    arr = np.asarray(arr, dtype=float)
                    arr = arr[np.isfinite(arr)]
                    if arr.size > 0:
                        y_all.append(arr)
                if not y_all:
                    return
                y    = np.concatenate(y_all)
                ymin = np.min(y)
                ymax = np.max(y)
                if np.isclose(ymin, ymax):
                    pad = max(abs(ymin) * 1e-06, 1e-12)
                else:
                    pad = (ymax - ymin) * pad_ratio
                ax.set_ylim(ymin - pad, ymax + pad)
                ax.yaxis.set_major_locator(MaxNLocator(nbins=4))
                if decimal:
                    ax.yaxis.set_major_formatter(FuncFormatter(lambda x, pos: f'{x:.2f}'))
                else:
                    ax.yaxis.set_major_formatter(FuncFormatter(lambda x, pos: f'{x:.2e}'))

            def set_safe_xlim(ax, start, *lists):
                """Set an x-axis range that remains valid for short histories."""
                max_len = max((len(lst) for lst in lists))
                if max_len <= start:
                    ax.set_xlim(start, start + 1)
                else:
                    ax.set_xlim(start, max_len - 1)

            def set_log_error_yaxis(ax, *arrays):
                """Set a logarithmic error axis with sparse decade labels."""
                y_all = []
                for arr in arrays:
                    arr = np.asarray(arr, dtype=float)
                    arr = arr[np.isfinite(arr)]
                    arr = arr[arr > 0.0]
                    if arr.size > 0:
                        y_all.append(arr)
                if not y_all:
                    return
                y        = np.concatenate(y_all)
                ymin     = np.min(y)
                ymax     = np.max(y)
                ymin_exp = int(np.floor(np.log10(ymin)))
                ymax_exp = int(np.ceil(np.log10(ymax)))
                ax.set_ylim(10.0 ** ymin_exp, 10.0 ** ymax_exp)
                labelled_exponents = np.arange(ymax_exp, ymin_exp - 1, -2)
                labelled_exponents = labelled_exponents[::-1]
                major_ticks        = 10.0 ** labelled_exponents
                ax.set_yticks(major_ticks)
                ax.yaxis.set_major_formatter(LogFormatterMathtext(base=10.0))
                ax.yaxis.set_minor_locator(NullLocator())
                ax.yaxis.set_minor_formatter(NullFormatter())
                ax.tick_params(axis='y', which='major', labelsize=9)
            x_std_greedy  = np.arange(1, len(self.GreedyErrorListStandart))
            x_quad_greedy = np.arange(1, len(self.GreedyErrorListQuadKernel))
            ax1.semilogy(x_std_greedy, self.GreedyErrorListStandart[1:], color=color_std, linestyle='-', linewidth=wid, label='standard kernel')
            ax1.semilogy(x_quad_greedy, self.GreedyErrorListQuadKernel[1:], color=color_quad, linestyle='--', linewidth=wid, label='product kernel')
            set_safe_xlim(ax1, 1, self.GreedyErrorListStandart, self.GreedyErrorListQuadKernel)
            ax1.set_title('Maximal GHJB residual $E_{\\mathrm{greedy}}^{\\mathrm{GHJB}}(n)$', pad=title_pad)
            ax1.set_xlabel('\\# greedy iterations', labelpad=xlabel_pad)
            ax1.set_ylabel('residual', labelpad=ylabel_pad)
            ax1.legend(loc='upper right', frameon=True)
            ax1.grid(True, which='major', alpha=0.45)
            ax1.grid(False, which='minor')
            x_std_true  = np.arange(len(self.trueErrorListStandart))
            x_quad_true = np.arange(len(self.trueErrorListQuadKernel))
            ax2.semilogy(x_std_true, self.trueErrorListStandart, color=color_std, linestyle='-', linewidth=wid)
            ax2.semilogy(x_quad_true, self.trueErrorListQuadKernel, color=color_quad, linestyle='--', linewidth=wid)
            set_safe_xlim(ax2, 0, self.trueErrorListStandart, self.trueErrorListQuadKernel, self.perfErrorListStandart, self.perfErrorListQuadKernel)
            set_log_error_yaxis(ax2, self.trueErrorListStandart, self.trueErrorListQuadKernel)
            ax2.set_title('Maximum pointwise error $E_{\\mathrm{test}}^{\\mathrm{val}}(\\eta,n)$', pad=title_pad)
            ax2.set_xlabel('\\# PI iterations', labelpad=xlabel_pad)
            ax2.set_ylabel('error', labelpad=ylabel_pad)
            ax2.grid(True, which='major', alpha=0.45)
            ax2.grid(False, which='minor')
            x_std_perf  = np.arange(len(self.perfErrorListStandart))
            x_quad_perf = np.arange(len(self.perfErrorListQuadKernel))
            ax5.semilogy(x_std_perf, self.perfErrorListStandart, color=color_std, linestyle='-', linewidth=wid)
            ax5.semilogy(x_quad_perf, self.perfErrorListQuadKernel, color=color_quad, linestyle='--', linewidth=wid)
            set_safe_xlim(ax5, 0, self.trueErrorListStandart, self.trueErrorListQuadKernel, self.perfErrorListStandart, self.perfErrorListQuadKernel)
            set_log_error_yaxis(ax5, self.perfErrorListStandart, self.perfErrorListQuadKernel)
            ax5.set_title('Maximum performance error $E_{\\mathrm{perf}}^{\\mathrm{val}}(\\eta,n)$', pad=title_pad)
            ax5.set_xlabel('\\# PI iterations', labelpad=xlabel_pad)
            ax5.set_ylabel('error', labelpad=ylabel_pad)
            ax5.grid(True, which='major', alpha=0.45)
            ax5.grid(False, which='minor')
            x_std_lower  = np.arange(len(self.lowerListStandart))
            x_quad_lower = np.arange(len(self.lowerListQuadKernel))
            y_std_lower  = np.asarray(self.lowerListStandart, dtype=float)
            y_quad_lower = np.asarray(self.lowerListQuadKernel, dtype=float)
            ax3.plot(x_std_lower, y_std_lower, color=color_std, linestyle='-', linewidth=wid)
            ax3.plot(x_quad_lower, y_quad_lower, color=color_quad, linestyle='--', linewidth=wid)
            ax3.axhline(y=lower, color=color_ref, linestyle=':', linewidth=1.4, label='$\\frac{1}{2}\\lambda_{\\min}(P_{AT})$')
            set_safe_xlim(ax3, 0, self.lowerListStandart, self.lowerListQuadKernel)
            set_zoomed_yaxis(ax3, y_std_lower, y_quad_lower, [lower], decimal=True)
            ax3.set_title('$\\mathrm{MinQuadraticBound}$', pad=title_pad)
            ax3.set_xlabel('\\# PI iterations', labelpad=xlabel_pad)
            ax3.set_ylabel('value', labelpad=ylabel_pad)
            ax3.legend(loc='center right', frameon=True)
            ax3.grid(True, which='major', alpha=0.45)
            ax3.grid(False, which='minor')
            x_std_upper  = np.arange(len(self.upperListStandart))
            x_quad_upper = np.arange(len(self.upperListQuadKernel))
            y_std_upper  = np.asarray(self.upperListStandart, dtype=float)
            y_quad_upper = np.asarray(self.upperListQuadKernel, dtype=float)
            ax4.plot(x_std_upper, y_std_upper, color=color_std, linestyle='-', linewidth=wid)
            ax4.plot(x_quad_upper, y_quad_upper, color=color_quad, linestyle='--', linewidth=wid)
            ax4.axhline(y=upper, color=color_ref, linestyle=':', linewidth=1.4, label='$2\\lambda_{\\max}(P_{AT})$')
            set_safe_xlim(ax4, 0, self.upperListStandart, self.upperListQuadKernel)
            set_zoomed_yaxis(ax4, y_std_upper, y_quad_upper, [upper], decimal=True)
            ax4.set_title('$\\mathrm{MaxQuadraticBound}$', pad=title_pad)
            ax4.set_xlabel('\\# PI iterations', labelpad=xlabel_pad)
            ax4.set_ylabel('value', labelpad=ylabel_pad)
            ax4.legend(loc='center right', frameon=True)
            ax4.grid(True, which='major', alpha=0.45)
            ax4.grid(False, which='minor')
            plt.savefig(f'{filename}.pdf', pad_inches=0.02, dpi=1200)
            plt.show()

    def plotObserver2(self, filename, plot_title, lower, upper):
        """Plot greedy, value, performance, and quadratic-bound errors for the heat-equation ROM."""
        import numpy as np
        import matplotlib.pyplot as plt
        from matplotlib.ticker import MaxNLocator, FuncFormatter, LogFormatterMathtext, NullFormatter, NullLocator, ScalarFormatter
        rc = {'text.usetex': True, 'font.family': 'serif', 'font.serif': ['Computer Modern Roman', 'CMU Serif', 'Latin Modern Roman', 'DejaVu Serif'], 'axes.titlesize': 15, 'axes.labelsize': 13, 'xtick.labelsize': 12, 'ytick.labelsize': 12, 'legend.fontsize': 15, 'lines.linewidth': 1.6, 'text.latex.preamble': '\\usepackage{amsmath,amssymb}'}
        with plt.rc_context(rc):
            wid        = 1.6
            color_std  = '#ff7f0e'
            color_quad = 'C2'
            color_ref  = 'black'
            fig        = plt.figure(figsize=(12, 6.8))
            fig.suptitle(plot_title, fontsize=20, y=0.955)
            left_margin       = 0.065
            right_margin      = 0.02
            column_gap        = 0.075
            column_width      = (1.0 - left_margin - right_margin - column_gap) / 2.0
            left_x            = left_margin
            right_x           = left_margin + column_width + column_gap
            title_pad         = 10
            xlabel_pad        = 6
            ylabel_pad        = 5
            vertical_gap_inch = 0.9
            vertical_gap      = vertical_gap_inch / fig.get_figheight()
            bottom_row_bottom = 0.09
            bottom_row_height = 0.13
            bottom_row_top    = bottom_row_bottom + bottom_row_height
            upper_bottom      = bottom_row_top + vertical_gap
            upper_top         = 0.84
            upper_height      = upper_top - upper_bottom
            ax1               = fig.add_axes([left_x, upper_bottom, column_width, upper_height])
            pos1              = ax1.get_position()
            right_gap         = vertical_gap
            right_plot_height = (pos1.height - right_gap) / 2.0
            ax2               = fig.add_axes([right_x, pos1.y1 - right_plot_height, column_width, right_plot_height])
            ax5               = fig.add_axes([right_x, pos1.y0, column_width, right_plot_height])
            ax3               = fig.add_axes([left_x, bottom_row_bottom, column_width, bottom_row_height])
            ax4               = fig.add_axes([right_x, bottom_row_bottom, column_width, bottom_row_height])

            def set_zoomed_yaxis(ax, *arrays, pad_ratio=0.15, decimal=False, sci_threshold_low=0.01, sci_threshold_high=10000.0):
                """Set a zoomed linear y-axis for finite plotting data."""
                y_all = []
                for arr in arrays:
                    arr = np.asarray(arr, dtype=float)
                    arr = arr[np.isfinite(arr)]
                    if arr.size > 0:
                        y_all.append(arr)
                if not y_all:
                    return
                y    = np.concatenate(y_all)
                ymin = np.min(y)
                ymax = np.max(y)
                if np.isclose(ymin, ymax):
                    pad = max(abs(ymin) * 1e-06, 1e-12)
                else:
                    pad = (ymax - ymin) * pad_ratio
                ax.set_ylim(ymin - pad, ymax + pad)
                ax.yaxis.set_major_locator(MaxNLocator(nbins=4))
                max_abs = np.max(np.abs(y))
                if decimal:
                    use_scientific = max_abs > 0.0 and (max_abs < sci_threshold_low or max_abs >= sci_threshold_high)
                    if use_scientific:
                        formatter = ScalarFormatter(useMathText=True)
                        formatter.set_scientific(True)
                        formatter.set_powerlimits((0, 0))
                        formatter.set_useOffset(False)
                        ax.yaxis.set_major_formatter(formatter)
                        ax.yaxis.get_offset_text().set_fontsize(9)
                    else:
                        ax.yaxis.set_major_formatter(FuncFormatter(lambda x, pos: f'{x:.2f}'))
                else:
                    ax.yaxis.set_major_formatter(FuncFormatter(lambda x, pos: f'{x:.2e}'))

            def set_safe_xlim(ax, start, *lists):
                """Set an x-axis range that remains valid for short histories."""
                max_len = max((len(lst) for lst in lists))
                if max_len <= start:
                    ax.set_xlim(start, start + 1)
                else:
                    ax.set_xlim(start, max_len - 1)

            def set_log_error_yaxis(ax, *arrays):
                """Set a logarithmic error axis with sparse decade labels."""
                y_all = []
                for arr in arrays:
                    arr = np.asarray(arr, dtype=float)
                    arr = arr[np.isfinite(arr)]
                    arr = arr[arr > 0.0]
                    if arr.size > 0:
                        y_all.append(arr)
                if not y_all:
                    return
                y        = np.concatenate(y_all)
                ymin     = np.min(y)
                ymax     = np.max(y)
                ymin_exp = int(np.floor(np.log10(ymin)))
                ymax_exp = int(np.ceil(np.log10(ymax)))
                ax.set_ylim(10.0 ** ymin_exp, 10.0 ** ymax_exp)
                labelled_exponents = np.arange(ymax_exp, ymin_exp - 1, -2)
                labelled_exponents = labelled_exponents[::-1]
                major_ticks        = 10.0 ** labelled_exponents
                ax.set_yticks(major_ticks)
                ax.yaxis.set_major_formatter(LogFormatterMathtext(base=10.0))
                ax.yaxis.set_minor_locator(NullLocator())
                ax.yaxis.set_minor_formatter(NullFormatter())
                ax.tick_params(axis='y', which='major', labelsize=9)
            x_std_greedy  = np.arange(1, len(self.GreedyErrorListStandart))
            x_quad_greedy = np.arange(1, len(self.GreedyErrorListQuadKernel))
            ax1.semilogy(x_std_greedy, self.GreedyErrorListStandart[1:], color=color_std, linestyle='-', linewidth=wid, label='standard kernel')
            ax1.semilogy(x_quad_greedy, self.GreedyErrorListQuadKernel[1:], color=color_quad, linestyle='--', linewidth=wid, label='product kernel')
            set_safe_xlim(ax1, 1, self.GreedyErrorListStandart, self.GreedyErrorListQuadKernel)
            ax1.set_title('Maximal GHJB residual $E_{\\mathrm{greedy}}^{\\mathrm{GHJB}}(n)$', pad=title_pad)
            ax1.set_xlabel('\\# greedy iterations', labelpad=xlabel_pad)
            ax1.set_ylabel('residual', labelpad=ylabel_pad)
            ax1.legend(loc='upper right', frameon=True)
            ax1.grid(True, which='major', alpha=0.45)
            ax1.grid(False, which='minor')
            x_std_true  = np.arange(len(self.trueErrorListStandart))
            x_quad_true = np.arange(len(self.trueErrorListQuadKernel))
            ax2.semilogy(x_std_true, self.trueErrorListStandart, color=color_std, linestyle='-', linewidth=wid)
            ax2.semilogy(x_quad_true, self.trueErrorListQuadKernel, color=color_quad, linestyle='--', linewidth=wid)
            set_safe_xlim(ax2, 0, self.trueErrorListStandart, self.trueErrorListQuadKernel, self.perfErrorListStandart, self.perfErrorListQuadKernel)
            set_log_error_yaxis(ax2, self.trueErrorListStandart, self.trueErrorListQuadKernel)
            ax2.set_title('Maximum pointwise error $E_{\\mathrm{test}}^{\\mathrm{val}}(\\eta,n)$', pad=title_pad)
            ax2.set_xlabel('\\# PI iterations', labelpad=xlabel_pad)
            ax2.set_ylabel('error', labelpad=ylabel_pad)
            ax2.grid(True, which='major', alpha=0.45)
            ax2.grid(False, which='minor')
            x_std_perf  = np.arange(len(self.perfErrorListStandart))
            x_quad_perf = np.arange(len(self.perfErrorListQuadKernel))
            ax5.semilogy(x_std_perf, self.perfErrorListStandart, color=color_std, linestyle='-', linewidth=wid)
            ax5.semilogy(x_quad_perf, self.perfErrorListQuadKernel, color=color_quad, linestyle='--', linewidth=wid)
            set_safe_xlim(ax5, 0, self.trueErrorListStandart, self.trueErrorListQuadKernel, self.perfErrorListStandart, self.perfErrorListQuadKernel)
            set_log_error_yaxis(ax5, self.perfErrorListStandart, self.perfErrorListQuadKernel)
            ax5.set_title('Maximum performance error $E_{\\mathrm{perf}}^{\\mathrm{val}}(\\eta,n)$', pad=title_pad)
            ax5.set_xlabel('\\# PI iterations', labelpad=xlabel_pad)
            ax5.set_ylabel('error', labelpad=ylabel_pad)
            ax5.grid(True, which='major', alpha=0.45)
            ax5.grid(False, which='minor')
            x_std_lower  = np.arange(len(self.lowerListStandart))
            x_quad_lower = np.arange(len(self.lowerListQuadKernel))
            y_std_lower  = np.asarray(self.lowerListStandart, dtype=float)
            y_quad_lower = np.asarray(self.lowerListQuadKernel, dtype=float)
            ax3.plot(x_std_lower, y_std_lower, color=color_std, linestyle='-', linewidth=wid)
            ax3.plot(x_quad_lower, y_quad_lower, color=color_quad, linestyle='--', linewidth=wid)
            ax3.axhline(y=lower, color=color_ref, linestyle=':', linewidth=1.4, label='$\\frac{1}{2}\\lambda_{\\min}(P_{HE})$')
            set_safe_xlim(ax3, 0, self.lowerListStandart, self.lowerListQuadKernel)
            set_zoomed_yaxis(ax3, y_std_lower, y_quad_lower, [lower], decimal=True)
            ax3.set_title('$\\mathrm{MinQuadraticBound}$', pad=title_pad)
            ax3.set_xlabel('\\# PI iterations', labelpad=xlabel_pad)
            ax3.set_ylabel('value', labelpad=ylabel_pad)
            ax3.legend(loc='center right', frameon=True)
            ax3.grid(True, which='major', alpha=0.45)
            ax3.grid(False, which='minor')
            x_std_upper  = np.arange(len(self.upperListStandart))
            x_quad_upper = np.arange(len(self.upperListQuadKernel))
            y_std_upper  = np.asarray(self.upperListStandart, dtype=float)
            y_quad_upper = np.asarray(self.upperListQuadKernel, dtype=float)
            ax4.plot(x_std_upper, y_std_upper, color=color_std, linestyle='-', linewidth=wid)
            ax4.plot(x_quad_upper, y_quad_upper, color=color_quad, linestyle='--', linewidth=wid)
            ax4.axhline(y=upper, color=color_ref, linestyle=':', linewidth=1.4, label='$2\\lambda_{\\max}(P_{HE})$')
            set_safe_xlim(ax4, 0, self.upperListStandart, self.upperListQuadKernel)
            set_zoomed_yaxis(ax4, y_std_upper, y_quad_upper, [upper], decimal=True)
            ax4.set_title('$\\mathrm{MaxQuadraticBound}$', pad=title_pad)
            ax4.set_xlabel('\\# PI iterations', labelpad=xlabel_pad)
            ax4.set_ylabel('value', labelpad=ylabel_pad)
            ax4.legend(loc='center right', frameon=True)
            ax4.grid(True, which='major', alpha=0.45)
            ax4.grid(False, which='minor')
            plt.savefig(f'{filename}.pdf', pad_inches=0.02, dpi=1200)
            plt.show()
