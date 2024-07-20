import plotly.express as px
import plotly.graph_objects as go


def plot_3d_points(data_index=0):
    # Get the points corresponding to the decoder layer and batch index
    points = array_data[data_index, 5, 0]

    # Initialize an empty list to store the traces
    traces = []

    # Loop through each query and plot
    for query_idx in range(points.shape[0]):
        x, y, z = (
            points[query_idx, :, 0],
            points[query_idx, :, 1],
            points[query_idx, :, 2],
        )

        trace = go.Scatter3d(
            x=x,
            y=y,
            z=z,
            mode="markers",
            marker=dict(
                size=2,
                opacity=0.8,
            ),
            name=f"Query {query_idx}",
        )
        traces.append(trace)

    # Create layout
    layout = go.Layout(
        margin=dict(l=2, r=2, b=2, t=2),
        scene=dict(
            aspectmode="data",
            camera=dict(
                eye=dict(x=2, y=2, z=2)  # Adjust these numbers to change viewpoint
            ),
        ),
    )

    # Create and show the 3D plot
    fig = go.Figure(data=traces, layout=layout)
    fig.show()


def show_3d():
    # Create layout
    layout = go.Layout(
        margin=dict(l=2, r=2, b=2, t=2),
        scene=dict(
            aspectmode="data",
            camera=dict(
                eye=dict(x=2, y=2, z=2)  # Adjust these numbers to change viewpoint
            ),
        ),
    )

    traces_lane3d = []
    for query_idx in range(flg.shape[0]):
        x, y, z = flg[query_idx, :, 0], flg[query_idx, :, 1], flg[query_idx, :, 2]

        trace = go.Scatter3d(
            x=x,
            y=y,
            z=z,
            mode="markers",
            marker=dict(
                size=2,
                opacity=0.8,
            ),
            name=f"bbox points {query_idx}",
        )
        traces_lane3d.append(trace)

    # Create and show the 3D plot
    fig = go.Figure(data=traces_lane3d, layout=layout)
    fig.show()

    # Initialize an empty list to store the traces
    traces = []

    # Loop through each query and plot
    for query_idx in range(fbg_p.shape[0]):
        x, y, z = fbg_p[query_idx, :, 0], fbg_p[query_idx, :, 1], fbg_p[query_idx, :, 2]

        trace = go.Scatter3d(
            x=x,
            y=y,
            z=z,
            mode="markers",
            marker=dict(
                size=2,
                opacity=0.8,
            ),
            name=f"bbox points {query_idx}",
        )
        traces.append(trace)

    trace_corners_all = []
    # i = 0
    for bbox in fbg:
        # bbox = annotation['bbox'][15]
        corners = get_8_corners_kitti(bbox).T[:, :3]  # Get the 8 corners

        trace_corners = go.Scatter3d(
            x=corners[:, 0],
            y=corners[:, 1],
            z=corners[:, 2],
            mode="markers",
            marker=dict(
                size=2,
                color="rgb(255, 0, 0)",  # set color to RGB values
                colorscale="Viridis",  # choose a colorscale
                opacity=0.8,
            ),
        )

        # Get X, Y and Z coordinates
        X, Y, Z = corners.T

        # Define the lines to connect vertices
        I = [0, 1, 2, 3, 0, 4, 5, 1, 5, 6, 2, 6, 7, 3, 7, 4]
        J = [1, 2, 3, 0, 4, 5, 6, 6, 1, 2, 3, 7, 7, 7, 4, 0]

        trace_line = go.Scatter3d(
            x=[X[i] for i in I + J],
            y=[Y[i] for i in I + J],
            z=[Z[i] for i in I + J],
            mode="lines",
            line=dict(color="rgb(125, 125, 125)"),
            hoverinfo="none",
        )
        trace_corners_all.append(trace_corners)
        trace_corners_all.append(trace_line)
        # break

    # Create and show the 3D plot
    fig = go.Figure(data=traces + trace_corners_all, layout=layout)
    fig.show()
