
import os 
import numpy as np
from dotenv import load_dotenv
import plotly.graph_objects as go
import chart_studio.plotly as py
import time

from qtpy.QtWidgets import QApplication, QMainWindow, QPushButton
from qtpy import QtCore, QtWidgets, QtWebEngineWidgets

from src.utils.InitializableDatasets import Dataset_NiiGz_3D_loadable


load_dotenv()

cube_triangles = (  
    [7, 0, 0, 0, 4, 4, 6, 6, 4, 0, 3, 2],
    [3, 4, 1, 2, 5, 6, 5, 2, 0, 1, 6, 3],
    [0, 7, 2, 3, 6, 7, 1, 1, 5, 5, 7, 6] )

def get_slices():
    width = height = 64
    n_slices = 52

    config = dict()
    config['input_size'] = (width, height, n_slices)
    dataset = Dataset_NiiGz_3D_loadable(
        image_path=os.getenv('IMAGE_PATH'),
        labels_path=os.getenv('LABEL_PATH'),
        config=config,
        slices=n_slices,
        slice_axis=2
    )
    data = dataset.get_dataset_index_information()
    img_name = "hippocampus_123.nii.gz"
    print(img_name)
    # get all images from a datapoint
    slices = np.empty(shape=(width, height, n_slices))
    labels = np.empty_like(slices)
    for i in range(n_slices):
        slices[..., i] = dataset[data[img_name][i]][1].squeeze()
        labels[..., i] = dataset[data[img_name][i]][2].squeeze()

    return slices, labels


def scatter_grid(size, elevation):
    X, Y = np.mgrid[0:size, 0:size]
    Z = np.ones_like(X)
    Z *= elevation

    x = X.flatten()
    y = Y.flatten()
    z = Z.flatten()
    return go.Scatter3d(x=x, y=y, z=z, mode='markers')

def unit_cube_vertices(x, y, z):
    size = 1
    target_x = x + size
    target_y = y + size
    target_z = z + size
    vert = (
        [x, x, target_x, target_x, x, x, target_x, target_x],
        [y, target_y, target_y, y, y, target_y, target_y, y],
        [z, z, z, z, target_z, target_z, target_z, target_z]
    )
    return vert

def simple_cube_mesh(x, y, z):

    v = unit_cube_vertices(x, y, z)

    cube = go.Mesh3d(
        x=v[0],
        y=v[1],
        z=v[2],
        i = cube_triangles[0],
        j = cube_triangles[1],
        k = cube_triangles[2],
        flatshading = True,
        color = "#d0fe1d",
        opacity=0.1,
        contour={"color": "black", "show": True},
        name=f"cube ({x}, {y}, {z})" 
    )
    return cube

def BIG_cube(size_x, size_y, size_z):

    data = []

    for i in range(size_x):
        for j in range(size_y):
            for k in range(size_z):
                data.append(simple_cube_mesh(i, j, k))

    return data

from qtplotly import PlotlyApplication

plotly_app = PlotlyApplication()

def sigmoid(x):
  return 1 / (1 + np.exp(-x))

@plotly_app.register("volume")
def volume():
    n = 52
    X, Y, Z = np.mgrid[0:64, 0:64, 0:n]
    vol, lbls = get_slices()
    vol = vol[..., :n]
    lbls = lbls[..., :n]
    volume = go.Volume(
        x=X.flatten(), y=Y.flatten(), z=Z.flatten(),
        value=vol.flatten(),
        opacity=0.3,
        surface_count=21,
        colorscale='gray',
        # opacityscale="max",#[[0.0, 0], [0.1, 0.5], [0.9, 1]],
        caps= dict(x_show=False, y_show=False, z_show=False),
    )
    print(lbls.shape)
    label_vol = go.Volume(
        x=X.flatten(), y=Y.flatten(), z=Z.flatten(),
        value=lbls.flatten(),
        opacity=0.8,
        opacityscale="max",
        surface_count=3,
        colorscale='solar',
    )

    f = np.load(os.path.join(os.getenv("PICKLE_PATH"), "temp.npy"), allow_pickle=True)
    ret = f[()]
    output = sigmoid(ret[20][..., 1])
    print(output.shape)
    output_vol = go.Volume(
        x=X.flatten(), y=Y.flatten(), z=Z.flatten(),
        value=output[:n].flatten(),
        opacity=0.5,
        opacityscale=[[0, 0.2], [0.5, 1], [1,
        0.2]],
        surface_count=1,
        colorscale='inferno',
    )

    labels = []
    # labels.append(simple_cube_mesh(32, 32, 15))

    data = [volume, label_vol, output_vol]
    # data += labels

    go.Frame

    fig = go.Figure(data=data)
    fig.update_scenes(aspectmode='data')
    return fig

@plotly_app.register("animation")
def animation_figure():
    n = 52
    X, Y, Z = np.mgrid[0:64, 0:64, 0:n]
    f = np.load(os.path.join(os.getenv("PICKLE_PATH"), "temp.npy"), allow_pickle=True)
    ret = f[()]

    vol, lbls = get_slices()
    vol = vol[..., :n]
    lbls = lbls[..., :n]
    volume = go.Volume(
        x=X.flatten(), y=Y.flatten(), z=Z.flatten(),
        value=vol.flatten(),
        opacity=0.3,
        surface_count=15,
        colorscale='gray',
        # opacityscale="max",#[[0.0, 0], [0.1, 0.5], [0.9, 1]],
        caps= dict(x_show=False, y_show=False, z_show=False),
    )
    print(f"vol: {vol.shape}")
    label_vol = go.Volume(
        x=X.flatten(), y=Y.flatten(), z=Z.flatten(),
        value=lbls.flatten(),
        opacity=0.8,
        opacityscale="max",
        surface_count=3,
        colorscale='solar',
    )

    def vol(i, name):
        output = sigmoid(ret[i][..., 1])
        return go.Volume(
            x=X.flatten(), y=Y.flatten(), z=Z.flatten(),
            value=output[:n].flatten(),
            opacity=0.5,
            opacityscale=[[0, 0.2], [0.5, 1], [1,
            0.2]],
            surface_count=1,
            colorscale='inferno',
            name=name,
            visible=False
        )

    # pause_btn = go.layout.updatemenu.Button(label="Pause", method="animate", args=[[None], {'frame': {'duration': 0, 'redraw': False}, 'mode': 'immediate',
    #             'transition': {'duration': 0}}])
    # play_btn = go.layout.updatemenu.Button(label="Play", method="animate", args=[None, dict(fromcurrent=True)])
    # update_menu = go.layout.Updatemenu(type="buttons", buttons=[pause_btn, play_btn], y=0.8)
    # slice_btns = [go.layout.updatemenu.Button(label=f"Slice {i}", method="animate", args=[[f"slice{i}"]]) for i in ret.keys()]
    # slice_menu = go.layout.Updatemenu(type="dropdown", buttons=slice_btns, showactive=True, y=1)
    # frames = [go.Frame(name=f"slice{x}", data=vol(x), layout=go.Layout(title=go.layout.Title(text=f"Slice: {x}", xanchor="center", x=0.5))) for x in ret.keys()]

    traces = [vol(i, f"slice{i}") for i in ret.keys()]

    figure = go.Figure()
    figure.add_trace(volume)
    figure.add_traces(traces)

    def active_one_hot(i):
        active = [False] * len(figure.data)
        active[i] = True
        return active
    

    slider = go.layout.Slider(
        active=10, 
        transition=go.layout.slider.Transition(duration=0),
        x=0,
        y=0,
        currentvalue=go.layout.slider.Currentvalue(font=go.layout.slider.currentvalue.Font(size=12), prefix=f"slice: ", visible=True, xanchor='center'),
        len=1.0,
        steps=[go.layout.slider.Step(
            method="update", #plotly function to call: https://plotly.com/javascript/plotlyjs-function-reference/#plotlyupdate
            args=[
                {"visible": active_one_hot(i)}, #changes to the traces
                {"title": "Slider switched to step: " + str(i)}, #changes to layout
                np.arange(1, len(figure.data)) #trace indices to update
            ] #args to plotly function
        ) for i in ret.keys()]
    )

    toggle_base_btn = go.layout.updatemenu.Button(label="Toggle Background", method="update", args=[{"visible": True}, None, [0]], args2=[{"visible": False}, None, [0]])
    button_menu = go.layout.Updatemenu(type="buttons", buttons=[toggle_base_btn], showactive=True, y=1)

    figure.update_layout(sliders=[slider])
    figure.update_layout(updatemenus=[button_menu])

    return figure

@plotly_app.register("tri")
def animation_figure():
    from skimage import measure

    n = 52
    X, Y, Z = np.mgrid[0:64, 0:64, 0:n]
    f = np.load(os.path.join(os.getenv("PICKLE_PATH"), "temp.npy"), allow_pickle=True)
    ret = f[()]
    net_out = ret[20][..., 1]
    vol, lbls = get_slices()

    verts, faces, _, _ = measure.marching_cubes(vol, allow_degenerate=True)
    x, y, z = zip(*verts)  
    I, J, K = faces.T
    tri = go.Mesh3d(
        x=x, y=y, z=z,
        i=I, j=J, k=K,
        opacity=0.3,
        colorscale='greys',
    )
    verts, faces, _, _ = measure.marching_cubes(lbls, allow_degenerate=True)
    x, y, z = zip(*verts)
    I, J, K = faces.T
    tri2 = go.Mesh3d(
        x=x, y=y, z=z,
        i=I, j=J, k=K,
        colorscale='plasma',
        opacity=0.3
    )

    def vol(i):
        net_out = sigmoid(ret[i][..., 1])
        verts, faces, _, _ = measure.marching_cubes(net_out, allow_degenerate=True)
        x, y, z = zip(*verts)
        I, J, K = faces.T
        tri1 = go.Mesh3d(
            x=x, y=y, z=z,
            i=I, j=J, k=K,
            opacity=0.4,
            colorscale='inferno',
            name=f"netout{i}",
            visible=False if i != 1 else True
        )
        return tri1

    traces = [vol(i) for i in list(ret.keys())[1:]]

    fig = go.Figure()
    fig.add_traces([tri, tri2])
    fig.add_traces(traces)

    def active_one_hot(i):
        active = [False] * len(traces)
        active[i] = True
        return active
    

    slider = go.layout.Slider(
        active=1, 
        transition=go.layout.slider.Transition(duration=0),
        x=0,
        y=0,
        currentvalue=go.layout.slider.Currentvalue(font=go.layout.slider.currentvalue.Font(size=12), prefix=f"slice: ", visible=True, xanchor='center'),
        len=1.0,
        steps=[go.layout.slider.Step(
            method="update", #plotly function to call: https://plotly.com/javascript/plotlyjs-function-reference/#plotlyupdate
            args=[
                {"visible": active_one_hot(i)}, #changes to the traces
                {"title": "Slider switched to step: " + str(i)}, #changes to layout
                np.arange(2, len(traces)+2) #trace indices to update
            ] #args to plotly function
        ) for i in range(len(traces))]
    )

    fig.update_layout(sliders=[slider])
    return fig


class Widget(QtWidgets.QWidget):

    def __init__(self, parent=None):
        super().__init__(parent)

        self.btn = QPushButton("Load")

        self.m_view = QtWebEngineWidgets.QWebEngineView()
        vlay = QtWidgets.QVBoxLayout(self)
        vlay.addWidget(self.btn)
        vlay.addWidget(self.m_view)

        self.m_view.loadStarted.connect(self.loadStartedHandler)
        self.m_view.loadProgress.connect(self.loadProgressHandler)
        self.m_view.loadFinished.connect(self.loadFinishedHandler)

        self.resize(1920, 1080)

    @QtCore.Slot()
    def loadStartedHandler(self):
        print(time.time(), ": load started")

    @QtCore.Slot(int)
    def loadProgressHandler(self, prog):
        self.btn.setText(f"Loading: {prog}")
        print(time.time(), ":load progress", prog)

    @QtCore.Slot()
    def loadFinishedHandler(self):
        self.btn.setText("Finished Loading")
        print(time.time(), ": load finished")

    @QtCore.Slot(str)
    def onCurrentIndexChanged(self, name):
        self.m_view.load(plotly_app.create_url(name))

if __name__ == "__main__":

    import sys
    app = QtWidgets.QApplication(sys.argv)
    plotly_app.init_handler()
    w = Widget()
    w.show()
    w.m_view.load(plotly_app.create_url("tri"))
    sys.exit(app.exec_())
    # animation_figure().show()
    # volume().show()

