from scipy.linalg import solve_triangular
import scipy.interpolate as interp
import numpy as np
import matplotlib.pylab as plt
from matplotlib.pyplot import cm
import imutils

import statsmodels.api as sm
#from statsmodels import api as sm

def rescale(arr):
    '''rescale array to [0,1]'''
    arr_min = arr.min()
    arr_max = arr.max()
    return (arr - arr_min) / (arr_max - arr_min)

def rerescale(arr, arr2):
    '''rescale array to scale of another array'''
    arr_min = arr.min()
    arr_max = arr.max()
    return arr2 * (arr_max - arr_min) + arr_min

def rotate_image(img_arr, center=None, angle=0):
    """
    Rotate image around center with keeping its size
    :param img_arr: original image numpy array
    :param center: center of image, two numbers
    :param angle: angle to rotate counterclockwise, degree
    :return: rotated numpy array image with the same shape as original
    """
    arr = np.zeros((img_arr.shape[0], img_arr.shape[1], 3))
    arr[:, :, 0] = img_arr.copy()
    # arr[:,:,0] = coeff * rescale(img_arr)
    # rotated = imutils.rotate(arr, angle, center=center)
    # return rerescale(arr, rotated[:, :, 0])/coeff
    return imutils.rotate(arr, angle, center=center)[:,:,0]


def generate_ell(r, incl):
    phi = np.linspace(0, np.pi, 100)
    cos_i = np.cos(incl * np.pi / 180.)

    x = np.cos(phi) * r
    y = np.array([cos_i * np.sqrt(r ** 2 - x_ ** 2) for x_ in x])

    x_ = np.cos(phi) * r
    y_ = -np.array([cos_i * np.sqrt(r ** 2 - x_ ** 2) for x_ in x])
    return np.concatenate([x, x_[::-1]]), np.concatenate([y, y_[::-1]])


def gen_ell_point(r, y, incl):
    cos_i = np.cos(incl * np.pi / 180.)
    return np.sqrt(r ** 2 - y ** 2 / cos_i ** 2)


def ell_line_inters(line_height, ell_height, cos_i, cen_x, cen_y):
    """
    Find two intersection points for line parallel X-axis and ellipsis
    :param line_height:
    :param ell_height:
    :param cos_i:
    :param cen_x:
    :param cen_y:
    :return:
    """
    yi = line_height
    yii = ell_height
    tmp = (yii / cos_i) ** 2 - (yi / cos_i) ** 2
    tmp = 0 if tmp < 1e-5 else tmp # TODO: why this is needed?
    right = [np.sqrt(tmp) + cen_x, yi + cen_y]
    left = [-np.sqrt(tmp) + cen_x, yi + cen_y]
    return (left, right)


def overlay_ellipses_grid(ax, ymin=None, delta_y=None, y_bins=None, X_max=None, center=None, incl=None):
    """
    Plot ellipses with step delta_y
    :param ax: image to plot on
    :param ymin: first ellipsis distance from center
    :param delta_y: ell width
    :param y_bins: number of ellipses
    :param X_max: tangent line boundary, from center
    :param center: (x,y) of center
    :param incl: inclination of galaxy
    """
    for ind_i in range(0, y_bins + 1, 1):
        cen_x,cen_y = center
        cos_i = np.cos(incl * np.pi / 180.)
        yi = ymin + ind_i * delta_y
        ax.plot([cen_x - X_max, cen_x + X_max], [cen_y + yi, cen_y + yi], '--', color='k', alpha=0.5)
        r0 = yi / cos_i
        x, y = generate_ell(r0,  incl)
        ax.plot(x + cen_x, y + cen_y, ':', color='k', alpha=0.5)


def plot_ifverbose_map(ax, map_data, title, lim_frame, ymin, delta_y, y_bins, X_max, cen_x, cen_y, **kwargs):
    """
    Plot data map.
    :param ax:
    :param map_data:
    :param title:
    :param lim_frame:
    :param ymin:
    :param delta_y:
    :param y_bins:
    :param X_max:
    :param cen_x:
    :param cen_y:
    :param kwargs:
    """
    im = ax.imshow(np.squeeze(map_data), origin='lower', **kwargs)
    if lim_frame:
        if ymin < 0 and delta_y < 0:
            ax.set_xlim(cen_x - X_max - 10, cen_x + X_max + 10)
            ax.set_ylim(cen_y + ymin + delta_y * y_bins - 10, cen_y + 5)
        else:
            ax.set_xlim(cen_x - X_max - 10, cen_x + X_max + 10)
            ax.set_ylim(cen_y - 5, cen_y + ymin + delta_y * y_bins + 10)
    plt.colorbar(im, fraction=0.046, pad=0.04, orientation="horizontal")
    ax.scatter(cen_x, cen_y, 50, 'k')
    ax.set_title(title)


def grid_point_val(points_grid, dmap=None, method='linear'):
    """
    This function takes 2d array and list of points and interpolate values to these points.
    :param points_grid: list of points to interpolate values into
    :param dmap: data map, 2d numpy array
    :param method: 'linear', 'nearest' or 'cubic'
    :return: list of interpolated values
    """
    xm, ym = np.meshgrid(np.arange(dmap.shape[1]), np.arange(dmap.shape[0]))
    return interp.griddata(list(zip(xm.ravel(), ym.ravel())), dmap.ravel(), points_grid, method=method)

def solve_TWR(intens=None,
              vel=None,
              center=None,
              incl=None,
              delta_y=15.,
              y_bins=4,
              X_max=None,
              ymin=None,
              verbose=False,
              r_scale=1.,
              lim_frame=False,
              ax=None,
              img_scale=1.5):
    """

    :param intens:
    :param vel:
    :param center:
    :param incl:
    :param delta_y: negative if ...
    :param y_bins:
    :param X_max:
    :param ymin: first ellipsis distance from center, `delta_y` if None
    :param verbose:
    :param r_scale:
    :param lim_frame:
    :param ax: TODO remove or rewrite
    :param img_scale: to scale verbose plots
    :return:
    """
    cen_x, cen_y = center

    cos_i = np.cos(incl * np.pi / 180.)
    sin_i = np.sin(incl * np.pi / 180.)

    if ymin is None:
        ymin = delta_y

    # plot velocity and intensity maps with superimposed ellipses
    if verbose:
        fig = plt.figure(figsize=[48 * img_scale, 24 * img_scale])

        ax = plt.subplot(121)
        plot_ifverbose_map(ax, vel, 'Velocity', lim_frame, ymin, delta_y, y_bins, X_max, cen_x, cen_y, cmap='rainbow', vmin=-200, vmax=200)
        overlay_ellipses_grid(ax, ymin=ymin, delta_y=delta_y, y_bins=y_bins, X_max=X_max, center=center, incl=incl)

        ax = plt.subplot(122)
        plot_ifverbose_map(ax, np.log2(intens), 'log2_Intensity', lim_frame, ymin, delta_y, y_bins, X_max, cen_x, cen_y, cmap='rainbow')
        overlay_ellipses_grid(ax, ymin=ymin, delta_y=delta_y, y_bins=y_bins, X_max=X_max, center=center, incl=incl)

        plt.show()

    # fill radial bins, where we will find Omega
    rrs = []
    for ind_i in range(0, y_bins + 1, 1):
        yi = ymin + ind_i * delta_y
        rrs.append(yi / cos_i)
    if verbose:
        print('Radial bins: {rrs}\n')

    # plot radial bins (right now images are the same, probably should replace second with something else)
    if verbose:

        fig = plt.figure(figsize=[48 * img_scale, 24 * img_scale])
        ax1 = plt.subplot(121)
        tmp = np.zeros(vel.shape)
        tmp[tmp == 0.0] = np.nan
        plot_ifverbose_map(ax1, tmp, 'Radial bins', lim_frame, ymin, delta_y, y_bins, X_max, cen_x, cen_y)
        overlay_ellipses_grid(ax1, ymin=ymin, delta_y=delta_y, y_bins=y_bins, X_max=X_max, center=center, incl=incl)

        bin_colors = cm.rainbow(np.linspace(0, 1, y_bins + 1))
        for ind_i in range(0, y_bins, 1):
            yi = ymin + ind_i * delta_y
            for ind_j in range(ind_i + 1, y_bins + 1, 1):
                yii = ymin + ind_j * delta_y
                pl, pr = ell_line_inters(yi, yii, cos_i, cen_x, cen_y)
                if ind_j > 1:
                    pl_, pr_ = ell_line_inters(yi, yii - delta_y, cos_i, cen_x, cen_y)
                else:
                    pl_, pr_ = (cen_x, yi + cen_y), (cen_x, yi + cen_y)

                color_ = bin_colors[ind_j]
                ax1.plot([pl[0], pl_[0]], [pl[1], pl_[1]], '-', color=color_)
                ax1.plot([pr[0], pr_[0]], [pr[1], pr_[1]], '-', color=color_)

        ax2 = plt.subplot(122)
        plot_ifverbose_map(ax2, tmp, 'Ellipses subparts', lim_frame, ymin, delta_y, y_bins, X_max, cen_x, cen_y)
        overlay_ellipses_grid(ax2, ymin=ymin, delta_y=delta_y, y_bins=y_bins, X_max=X_max, center=center, incl=incl)

        bin_colors = cm.rainbow(np.linspace(0, 1, y_bins + 1))
        for ind_i in range(0, y_bins, 1):
            yi = ymin + ind_i * delta_y
            for ind_j in range(ind_i + 1, y_bins + 1, 1):
                yii = ymin + ind_j * delta_y
                pl, pr = ell_line_inters(yi, yii, cos_i, cen_x, cen_y)
                if ind_j > 1:
                    pl_, pr_ = ell_line_inters(yi, yii - delta_y, cos_i, cen_x, cen_y)
                else:
                    pl_, pr_ = (cen_x, yi + cen_y), (cen_x, yi + cen_y)

                color_ = bin_colors[ind_j]
                ax2.plot([pl[0], pl_[0]], [pl[1], pl_[1]], '|-', color=color_)
                ax2.plot([pr[0], pr_[0]], [pr[1], pr_[1]], '|-', color=color_)

        plt.show()

    # function to plot boundaries of ellipses
    def plot_vert_lines(ind, ax):
        ax.axvline(x=cen_x, color='k', alpha=0.5, ls='--')
        for ind_j in range(ind + 1, y_bins + 1, 1):
            yii = ymin + ind_j * delta_y
            xxx_l, xxx_r = ell_line_inters(yi, yii, cos_i, cen_x, cen_y)
            ax.axvline(x=xxx_r[0], color='k', alpha=0.5, ls=':')
            ax.axvline(x=xxx_l[0], color='k', alpha=0.5, ls=':')

    # triangular matrix left and right sides, see paper for details
    K = np.zeros(shape=(y_bins, y_bins))
    b = np.zeros(y_bins)

    # fill equations (triangular matrix)
    for ind_i in range(0, y_bins, 1):
        yi = ymin + ind_i * delta_y

        # plot individual slices
        if verbose:
            fig, axes = plt.subplots(figsize=[30*img_scale, 14*img_scale], nrows=2, ncols=2, sharex=True)

            # this slice used actual pixels on line, not interpolated values
            int_slice = int(cen_y + yi + 1)
            xmi,xma = int(cen_x - X_max - 1), int(cen_x + X_max + 1)
            xrange = range(xmi,xma, 1)

            ax = axes[0,0]
            vdata = np.ravel(vel[int_slice, xmi:xma])
            ax.plot(xrange, vdata, '.-', color='r')
            ax.set_title('Velocity: slice yi={:2.2f} [pix] from center'.format(yi))
            plot_vert_lines(ind_i, ax)

            ax = axes[1, 0]
            idata = np.ravel(intens[int_slice, xmi:xma])
            ax.plot(xrange, idata, '.-', color='b')
            ax.set_title('Intens: slice yi={:2.2f} [pix] from center'.format(yi))
            plot_vert_lines(ind_i, ax)

        # here we interpolate data to new grid;
        # We probably can use exact pixels on line, but this can be difficult in case of y between pixels, etc.
        bin_colors = cm.rainbow(np.linspace(0, 1, y_bins + 1))
        for ind_j in range(ind_i + 1, y_bins + 1, 1):
            yii = ymin + ind_j * delta_y
            pl, pr = ell_line_inters(yi, yii, cos_i, cen_x, cen_y)
            if ind_j > 1:
                pl_, pr_ = ell_line_inters(yi, yii - delta_y, cos_i, cen_x, cen_y)
            else:
                pl_, pr_ = (cen_x, yi + cen_y), (cen_x, yi + cen_y)

            color_ = bin_colors[ind_j]

            full_slice_points_r = [(_, yi + cen_y + 0.5) for _ in np.arange(pr_[0], pr[0], 1)]
            sigma_rr = grid_point_val(full_slice_points_r, dmap=intens)
            vel_rr = grid_point_val(full_slice_points_r, dmap=vel)

            # plot that interpolated values are good and correct
            if verbose:
                ax.plot([_[0] for _ in full_slice_points_r], sigma_rr, 'x', color=color_)
                axes[0,0].plot([_[0] for _ in full_slice_points_r], vel_rr, 'x', color=color_)

            full_slice_points_l = [(cen_x - (_ - cen_x), yi + cen_y + 0.5) for _ in np.arange(pr_[0], pr[0], 1)]
            sigma_ll = grid_point_val(full_slice_points_l, dmap=intens)
            vel_ll = grid_point_val(full_slice_points_l, dmap=vel)

            if verbose:
                ax.plot([_[0] for _ in full_slice_points_l], sigma_ll, 'x', color=color_)
                axes[0, 0].plot([_[0] for _ in full_slice_points_l], vel_ll, 'x', color=color_)

            # right side of equation
            mult_rr = sigma_rr * vel_rr / sin_i
            mult_ll = sigma_ll * vel_ll / sin_i
            b[ind_i] += np.sum(mult_rr + mult_ll)

            # plot it for each segment with color selected accordingly
            if verbose:
                axes[1, 1].fill_between([_[0] for _ in full_slice_points_l], [0] * len(mult_ll), mult_ll, color=color_)
                axes[1, 1].fill_between([_[0] for _ in full_slice_points_r], [0] * len(mult_rr), mult_rr, color=color_)

            xsigma_r = (np.array([_[0] for _ in full_slice_points_r]) - cen_x) * sigma_rr
            xsigma_l = (np.array([_[0] for _ in full_slice_points_l]) - cen_x) * sigma_ll

            if verbose:
                axes[0, 1].fill_between([_[0] for _ in full_slice_points_r], [0] * len(xsigma_r), xsigma_r, color=color_)
                axes[0, 1].fill_between([_[0] for _ in full_slice_points_l], [0] * len(xsigma_l), xsigma_l, color=color_)

            delta_Sigma = xsigma_r + xsigma_l # plus is because delta_x for left side is negative, see plots
            K[ind_i, ind_j-1] = np.sum(delta_Sigma) * r_scale  # once because dx in pix on the both sides

        if verbose:
            axes[1, 1].set_title('Bi: vel * intens / sin_i')
            axes[0, 1].set_title('K: intens * x')
            plot_vert_lines(ind_i, axes[0, 1])
            plot_vert_lines(ind_i, axes[1, 1])
    if verbose:
        print('K:\n ', K)
        print('b:\n ', b)
    try:
        # solve equations and print results
        om = solve_triangular(K, b)
        if verbose:
            print('Omega:', om)
            print('Residuals:', b - K.dot(om))
        return om, rrs
    except Exception as e:
        # TODO: add handling
        # print(e)
        # print('===========DEBUG===============')
        # print('Check individual elements of matrix:\n')
        # ccount = 0
        # for ind_i in range(0, y_bins, 1):
        #     for ind_j in range(ind_i, y_bins, 1):
        #         r = rrs[ind_j + 1] * r_scale
        #         delta_Sigma = sigma_r[ccount] - sigma_l[ccount]
        #         if r == 0:
        #             print(f'K[{ind_i},{ind_j}] = 0 because rrs[{ccount}]={r}')
        #         if delta_Sigma == 0:
        #             print(f'K[{ind_i},{ind_j}] = 0 because delta_Sigma[{ccount}]={delta_Sigma}')
        #             print('(sigma_r = {}, sigma_l = {})'.format(sigma_r[ccount], sigma_l[ccount]))
        #         ccount += 1
        # print('===========DEBUG END===============')
        return


def solve_TW(intens=None,
              vel=None,
              center=None,
              incl=None,
              ys = [],
              delta_y=15.,
              y_bins=4,
              X_max=None,
              ymin=0.,
              verbose=True,
              r_scale=1.,
              ax=None,
              img_scale=1.5,
              x1=-3,
              x2=3,
              y1=-150,
              y2=150):

    cen_x, cen_y = center

    cos_i = np.cos(incl * np.pi / 180.)
    sin_i = np.sin(incl * np.pi / 180.)

    if len(ys) == 0:
        ys = [ymin+_*delta_y for _ in range(-y_bins, y_bins+1, 1)]

    cen_x, cen_y = center

    cos_i = np.cos(incl * np.pi / 180.)
    sin_i = np.sin(incl * np.pi / 180.)

    if ymin is None:
        ymin = delta_y

    # plot velocity and intensity maps along with apertures
    if verbose:
        fig = plt.figure(figsize=[48 * img_scale, 24 * img_scale])

        ax = plt.subplot(121)
        plot_ifverbose_map(ax, vel, 'Velocity', False, ymin, delta_y, y_bins, X_max, cen_x, cen_y, cmap='rainbow',
                           vmin=-200, vmax=200)

        for yi in ys:
            ax.plot([cen_x - X_max, cen_x + X_max], [cen_y + yi, cen_y + yi], '--', color='k', alpha=0.5)

        ax = plt.subplot(122)
        plot_ifverbose_map(ax, np.log2(intens), 'log2_Intensity', False, ymin, delta_y, y_bins, X_max, cen_x, cen_y,
                           cmap='rainbow', vmin=-2)

        for yi in ys:
            ax.plot([cen_x - X_max, cen_x + X_max], [cen_y + yi, cen_y + yi], '--', color='k', alpha=0.5)

        plt.show()

    right_side = np.zeros(len(ys))
    left_side = np.zeros(len(ys))
    intens_integral = np.zeros(len(ys))

    # fill equations right and left side
    for ind_i, yi in enumerate(ys):
        full_slice_points = [(_ + cen_x + 0.5, yi + cen_y + 0.5) for _ in np.arange(-X_max, X_max, 1)]

        sigma_fsp = grid_point_val(full_slice_points, dmap=intens)
        vel_fsp = grid_point_val(full_slice_points, dmap=vel)
        right_side[ind_i] = np.sum(sigma_fsp * vel_fsp / sin_i)

        x_fsp = np.array([_[0] * r_scale for _ in full_slice_points]) - cen_x * r_scale
        left_side[ind_i] = np.sum(sigma_fsp * x_fsp)

        intens_integral[ind_i] = np.sum(sigma_fsp)


    # TODO: old integrer version without interpolation, do smth with it
    # for ind_i in range(0, y_bins, 1):
    #     yi = ymin + ind_i * delta_y
    #
    #     int_slice = int(cen_y + yi + 1)
    #     xmi, xma = int(cen_x - X_max - 1), int(cen_x + X_max + 1)
    #     xrange = range(xmi, xma, 1)
    #
    #     vdata = np.ravel(vel[int_slice, xmi:xma])
    #     idata = np.ravel(intens[int_slice, xmi:xma])
    #
    #     right_side[ind_i] = np.sum(idata * vdata / sin_i)
    #
    #     x_fsp = (np.array(xrange) - cen_x) * r_scale
    #     left_side[ind_i] = np.sum(idata * x_fsp)
    #
    #     intens_integral[ind_i] = np.sum(idata)

    # calculate <x> and <v_y> values weighted by intensity integral
    xp, yp = np.array(left_side) / np.array(intens_integral), np.array(right_side) / np.array(intens_integral)

    # build model
    x = sm.add_constant(xp.reshape((-1, 1)))
    ols = sm.OLS(yp, x)
    ols_result = ols.fit()

    if verbose:

        r_sq = ols_result.rsquared
        print('coefficient of determination R^2:', r_sq)
        print('intercept: {:.3f}+/-{:.3f} (SE)'.format(ols_result.params[0], ols_result.bse[0]))
        print('slope Ω: {:.3f}+/-{:.3f} (SE)'.format(ols_result.params[1], ols_result.bse[1]))

        fig, axes = plt.subplots(figsize=[14 * img_scale, 7 * img_scale], ncols=2, nrows=1, sharey=True, sharex=True)

        ax = axes[0]
        ax.plot(xp, yp, 'x--', color='b')
        for ind_i, yi in enumerate(ys):
            ax.annotate(str(yi), (xp[ind_i], yp[ind_i]))

        ax.set_xlabel('<x>, kpc')
        ax.set_ylabel('<V>, km/s')

        ax = axes[1]
        ax.plot(xp, yp, 'o', color='b')

        def linm(slope, intercept, xvals):
            return slope * xvals + intercept

        pps = np.linspace(min(xp)*0.9, max(xp)*1.1, 100)

        y_pred = linm(ols_result.params[1], ols_result.params[0], pps)
        plt.plot(pps, y_pred, '--', color='r')
        y_pred_max = linm(ols_result.params[1] - ols_result.bse[1], ols_result.params[0], pps)
        y_pred_min = linm(ols_result.params[1] + ols_result.bse[1], ols_result.params[0], pps)
        plt.fill_between(pps, y_pred_max, y_pred_min, alpha=0.3, color='r')

        plt.title('slope Ω: {:.3f}+/-{:.3f} (SE)'.format(ols_result.params[1], ols_result.bse[1]))

        ax.set_xlabel('<x>, kpc')
        ax.set_ylabel('<V>, km/s')
        ax.set_xlim(x1,x2)
        ax.set_ylim(y1,y2)
        plt.show()
    omega, omega_se = ols_result.params[1], ols_result.bse[1]

    return omega, omega_se, xp, yp
