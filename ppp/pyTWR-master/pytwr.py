from scipy.linalg import solve_triangular
import scipy.interpolate as interp
import numpy as np
import matplotlib.pylab as plt
import imutils

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

def prepare_maps(PA, incl, vel, intens):
    # TODO
    pass

def add_ells(ax, ymin=None, delta_y=None, y_bins=None, X_max=None, center=None, incl=None):
    for ind_i in range(0, y_bins + 1, 1):
        cen_x,cen_y = center
        cos_i = np.cos(incl * np.pi / 180.)
        yi = ymin + ind_i * delta_y
        ax.plot([cen_x - X_max, cen_x + X_max], [cen_y + yi, cen_y + yi], '--', color='k', alpha=0.5)
        r0 = yi / cos_i
        x, y = generate_ell(r0,  incl)
        ax.plot(x + cen_x, y + cen_y, ':', color='k', alpha=0.5)


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
              # intens_points_only=False,
              ax=None):

    cen_x, cen_y = center

    cos_i = np.cos(incl * np.pi / 180.)
    sin_i = np.sin(incl * np.pi / 180.)

    if ymin is None:
        ymin = delta_y

    if verbose:
        if ax is None:
            fig = plt.figure(figsize=[48 * 1.5, 24 * 1.5])
            ax = plt.subplot(121)
        im = ax.imshow(np.squeeze(vel), origin='lower', cmap='rainbow', vmin=-200, vmax=200)
        if lim_frame:
            if ymin < 0 and delta_y < 0:
                ax.set_xlim(cen_x - X_max - 10, cen_x + X_max + 10)
                ax.set_ylim(cen_y + ymin + delta_y * y_bins - 10, cen_y + 5)
            else:
                ax.set_xlim(cen_x - X_max - 10, cen_x + X_max + 10)
                ax.set_ylim(cen_y - 5, cen_y + ymin + delta_y * y_bins + 10)
        plt.colorbar(im, fraction=0.046, pad=0.04, orientation="horizontal")

        #         divider = make_axes_locatable(ax)
        #         cax = divider.append_axes("right", size="5%", pad=0.1)
        #         plt.colorbar(im, cax=cax)

        ax.scatter(cen_x, cen_y, 50, 'k')
        ax.set_title('velocity')

    for ind_i in range(0, y_bins + 1, 1):
        yi = ymin + ind_i * delta_y
        if verbose:
            ax.plot([cen_x - X_max, cen_x + X_max], [cen_y + yi, cen_y + yi], '--', color='r', alpha=0.5)
        r0 = yi / cos_i
        x, y = generate_ell(r0,  incl)
        if verbose:
            ax.plot(x + cen_x, y + cen_y, ':', color='r', alpha=0.5)

    if verbose:
        ax = plt.subplot(122)
        im = plt.imshow(np.squeeze(np.log2(intens)), origin='lower', cmap='rainbow')
        if lim_frame:
            if ymin < 0 and delta_y < 0:
                ax.set_xlim(cen_x - X_max - 10, cen_x + X_max + 10)
                ax.set_ylim(cen_y + ymin + delta_y * y_bins - 10, cen_y + 5)
            else:
                ax.set_xlim(cen_x - X_max - 10, cen_x + X_max + 10)
                ax.set_ylim(cen_y - 5, cen_y + ymin + delta_y * y_bins + 10)
        plt.colorbar(im, fraction=0.046, pad=0.04, orientation="horizontal")
        ax.scatter(cen_x, cen_y, 50, 'k')
        ax.set_title('log2_Sigma')

    # fill radial bins
    rrs = []
    for ind_i in range(0, y_bins + 1, 1):
        yi = ymin + ind_i * delta_y
        if verbose:
            ax.plot([cen_x - X_max, cen_x + X_max], [cen_y + yi, cen_y + yi], '--', color='k', alpha=0.5)
        r0 = yi / cos_i
        rrs.append(r0)
        x, y = generate_ell(r0, incl)
        if verbose:
            ax.plot(x + cen_x, y + cen_y, ':', color='k', alpha=0.5)

    if verbose:
        print(f'radial bins: {rrs}\n')
        for tmp_rrs in rrs[::-1]:
            ax.plot([cen_x, cen_x + tmp_rrs], [cen_y, cen_y], '.-')

    points_r = []
    points_l = []
    for ind_i in range(0, y_bins, 1):
        yi = ymin + ind_i * delta_y
        for ind_j in range(ind_i + 1, y_bins + 1, 1):
            yii = ymin + ind_j * delta_y
            points_r.append([np.sqrt((yii / cos_i) ** 2 - (yi / cos_i) ** 2) + cen_x, yi + cen_y])
            points_l.append([-np.sqrt((yii / cos_i) ** 2 - (yi / cos_i) ** 2) + cen_x, yi + cen_y])

    if verbose:
        style = dict(size=10, color='gray')

        for tmp_ind, point in enumerate(points_r):
            ax.scatter(point[0], point[1], 10., marker='s', color='m')
            ax.text(point[0], point[1], str(tmp_ind), ha='right', **style)

        for tmp_ind, point in enumerate(points_l):
            ax.scatter(point[0], point[1], 10., marker='s', color='c')
            ax.text(point[0], point[1], str(tmp_ind), ha='right', **style)

    def grid_point_val(pp, dmap=None):
        xm, ym = np.meshgrid(np.arange(dmap.shape[1]), np.arange(dmap.shape[0]))
        return interp.griddata(list(zip(xm.ravel(), ym.ravel())), dmap.ravel(), pp, method='linear')

    sigma_r = grid_point_val(points_r, dmap=intens)
    sigma_l = grid_point_val(points_l, dmap=intens)

    K = np.zeros(shape=(y_bins, y_bins))
    b = np.zeros(y_bins)

    # TODO: check; we do not multiply with r_scale because dx also will be in pix
    dr = delta_y / cos_i

    ccount = 0
    for ind_i in range(0, y_bins, 1):
        yi = ymin + ind_i * delta_y

        if False:
            pass
        # if intens_points_only:
        #     slice_points = [(_, cen_y + yi) for _ in np.arange(cen_x - X_max, cen_x + X_max, 1.)]
        #     # TODO: implement
        #     xp = []
        #     raise NotImplementedError('Wrong `intens_points_only`, buddy.')
        else:
            slice_points = []
            xp = [cen_x]
            for ind_j in range(ind_i + 1, y_bins + 1, 1):
                yii = ymin + ind_j * delta_y
                xp.append(np.sqrt((yii / cos_i) ** 2 - (yi / cos_i) ** 2) + cen_x)

                slice_points.append([np.sqrt((yii / cos_i) ** 2 - (yi / cos_i) ** 2) + cen_x, yi + cen_y])
                slice_points.append([-np.sqrt((yii / cos_i) ** 2 - (yi / cos_i) ** 2) + cen_x, yi + cen_y])

        #         if verbose:
        #             for iiii in range(0, len(xp)-1):
        #                 start_ = xp[iiii]
        #                 finish_ = xp[iiii+1]
        #                 ax.plot([start_, finish_], [yi+cen_y, yi+cen_y], '.-')

        full_slice_points = [(_ + cen_x, yi + cen_y) for _ in np.arange(-X_max - 0.5, X_max + 0.5, 1)]
        #         if verbose:
        #             ax.scatter([_[0] for _ in full_slice_points], [_[1] for _ in full_slice_points], 10, marker='x')

        v_y = grid_point_val(full_slice_points, dmap=vel) / sin_i
        sigma = grid_point_val(full_slice_points, dmap=intens)

        #         if verbose:
        #             print(v_y)
        #             print(sigma)

        #         if verbose:
        #             print(f'i:{ind_i}\n v_y:{v_y}\n sigma:{sigma}\n dx:{dx}')

        b[ind_i] = np.sum(v_y * sigma)

        diff_xp = list(np.diff(xp))

        for ind_j in range(ind_i, y_bins, 1):

            stx = xp[ind_j - ind_i]
            #             enx = xp[ind_j-ind_i+1]+1
            enx = xp[ind_j - ind_i + 1]

            #             print('!!!!!stxenx', stx, enx)
            #             print('!!!!!arange', np.arange(stx-0.5, enx+0.5, 1) - xp[0])

            full_slice_points_ = [(_, yi + cen_y) for _ in np.arange(stx, enx, 1)]
            sigma_rr = grid_point_val(full_slice_points_, dmap=intens)

            #             print('!!!!!fsp', full_slice_points_)
            #             print('!!!!!ssrr', sigma_rr)

            full_slice_points_ = [(cen_x - (_ - cen_x), yi + cen_y) for _ in np.arange(stx, enx, 1)]
            sigma_ll = grid_point_val(full_slice_points_, dmap=intens)

            if verbose:
                ax.scatter([_[0] for _ in full_slice_points_], [_[1] for _ in full_slice_points_], 10, marker='o',
                           color='k', alpha=0.3)

            #             print('!!!!!fsp', full_slice_points_)
            #             print('!!!!!ssll', sigma_ll)

            #             r = rrs[ind_j+1]*r_scale
            delta_Sigma = sigma_rr - sigma_ll
            #             K[ind_i, ind_j] = r*delta_Sigma*dr
            K[ind_i, ind_j] = np.sum(
                delta_Sigma * (np.arange(stx, enx, 1) - xp[0])) * r_scale  # once because dx in pix on the both sides

            if verbose:
                ddist = np.arange(stx, enx, 1) - xp[0]
                ax.plot([cen_x, cen_x + ddist[-1]], [yi + cen_y, yi + cen_y], '-', color='k')

            #             print(i,j,K[i,j],r,Sigma, sigma_r[ccount], sigma_l[ccount],ccount)
            ccount += 1

    if verbose:
        print('K:\n ', K)
        print('b:\n ', b)
    try:
        om = solve_triangular(K, b)
        if verbose:
            print('Omega:', om)
            print('Residuals:', b - K.dot(om))
        return om, rrs
    except Exception as e:
        print(e)
        print('===========DEBUG===============')
        print('Check individual elements of matrix:\n')
        ccount = 0
        for ind_i in range(0, y_bins, 1):
            for ind_j in range(ind_i, y_bins, 1):
                r = rrs[ind_j + 1] * r_scale
                delta_Sigma = sigma_r[ccount] - sigma_l[ccount]
                if r == 0:
                    print(f'K[{ind_i},{ind_j}] = 0 because rrs[{ccount}]={r}')
                if delta_Sigma == 0:
                    print(f'K[{ind_i},{ind_j}] = 0 because delta_Sigma[{ccount}]={delta_Sigma}')
                    print('(sigma_r = {}, sigma_l = {})'.format(sigma_r[ccount], sigma_l[ccount]))
                ccount += 1
        print('===========DEBUG END===============')
        return
