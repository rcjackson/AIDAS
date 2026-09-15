Batch processing
================
AIDAS supports batch inference so that you can infer the location of the lake breeze over
multiple radar scans with one inference step. Batch inference will save computational time
by loading the model only once and utilizing vectorization to perform the inference on multiple
radar scans at a time.

.. code-block :: python

    import aidas

    # List of radar scans to process
    radar_scans = [
        ('KLOT', '2025-07-15T18:00:00'),
        ('KLOT', '2025-07-15T19:00:00'),
        ('KLOT', '2025-07-15T20:00:00'),
    ]

    # Preprocess all radar scans
    preprocessed_scans = [aidas.io.preprocess_radar_image(station, time) for station, time in radar_scans]

    # Perform batch inference
    lake_breeze_results = aidas.model.infer_lake_breeze_batch(
        preprocessed_scans, model_name='lakebreeze_model_fcn_resnet50_no_augmentation')

   
Analyzing the mask data in custom workflows
-------------------------------------------
The :py:class:`~aidas.io.RadarImage` class contains all of the information you need to perform
custom analyses of the lake breeze front. 