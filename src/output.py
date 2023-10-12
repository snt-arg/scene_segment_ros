from fastsam.utils import convert_box_xywh_to_xyxy


def fastSamVisualizer(masks, pointPrompt, boxPrompts, pointLabel, counters):
    """
    Shows the output segmented image

    Parameters
    -------
    masks: Any
        List of masks provided by the segmenter

    Returns
    -------
    result: Mat
        The segmented visualized image
    """
    # Init
    bboxes = None
    points = None
    textPrompt = None
    # Read values
    boxPrompt = convert_box_xywh_to_xyxy(boxPrompts)
    # Annotations
    if boxPrompt[0][2] != 0 and boxPrompt[0][3] != 0:
        ann = masks.box_prompt(bboxes=boxPrompt)
        bboxes = boxPrompt
    elif textPrompt != None:
        ann = masks.text_prompt(text=textPrompt)
    elif pointPrompt[0] != [0, 0]:
        ann = masks.point_prompt(
            points=pointPrompt, pointlabel=pointLabel
        )
        points = pointPrompt
    else:
        ann = masks.everything_prompt()
    # Plotting
    result = masks.plot(
        annotations=ann,
        output_path='',
        bboxes=bboxes,
        points=points,
        point_label=pointLabel,
        withContours=counters,
        better_quality=False,
    )
    # Return
    return result
