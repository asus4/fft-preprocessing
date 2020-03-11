using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using Unity.Collections;
using Unity.Jobs;
using Unity.Mathematics;

[RequireComponent(typeof(AudioSource))]
public class DFTPreprocessRealTime : MonoBehaviour
{
    [SerializeField, Range(1f, 100f)] float _range = 60;
    [SerializeField] RectTransform _bypassMeter = null;
    [SerializeField] RectTransform _lowPassMeter = null;
    [SerializeField] RectTransform _bandPassMeter = null;
    [SerializeField] RectTransform _highPassMeter = null;

    [SerializeField, Range(50f, 5000f)] float _bandPassFreq = 960f;
    [SerializeField, Range(0.1f, 10f)] float _bandPassQ = 1.5f;


    AudioSource _source;
    MultibandFilter _filter;

    float[] _rawSamples = new float[1024];

    [Unity.Burst.BurstCompile(CompileSynchronously = true)]
    struct FilterRmsJob : IJob
    {
        [ReadOnly] public NativeSlice<float> Input;
        [WriteOnly] public NativeArray<float4> Output;
        public NativeArray<MultibandFilter> Filter;

        public void Execute()
        {
            var filter = Filter[0];

            // Square sum
            var ss = float4.zero;

            for (var i = 0; i < Input.Length; i++)
            {
                var vf = filter.FeedSample(Input[i]);
                ss += vf * vf;
            }

            // Root mean square
            var rms = math.sqrt(ss / Input.Length);

            // RMS in dBFS
            // Full scale sin wave = 0 dBFS : refLevel = 1/sqrt(2)
            const float refLevel = 0.7071f;
            const float zeroOffset = 1.5849e-13f;
            var level = 20 * math.log10(rms / refLevel + zeroOffset);

            // Output
            Output[0] = level;
            Filter[0] = filter;
        }
    }

    private void Start()
    {
        _source = GetComponent<AudioSource>();
    }


    private void Update()
    {
        // Single element native array used to share structs with the job.
        var tempFilter = new NativeArray<MultibandFilter>
          (1, Allocator.TempJob, NativeArrayOptions.UninitializedMemory);

        var tempLevel = new NativeArray<float4>
          (1, Allocator.TempJob, NativeArrayOptions.UninitializedMemory);

        // Filter update
        float sampleRate = _source.clip.frequency;
        _filter.SetParameter(_bandPassFreq / sampleRate, _bandPassQ);
        tempFilter[0] = _filter;

        _source.GetOutputData(_rawSamples, 0);
        var samples = new NativeArray<float>(_rawSamples, Allocator.TempJob);
        var sliceSamples = new NativeSlice<float>(samples);


        // Run the job on the main thread.
        new FilterRmsJob
        {
            Input = sliceSamples,
            Filter = tempFilter,
            Output = tempLevel
        }.Run();

        // Preserve the filter state.
        _filter = tempFilter[0];

        // Meter scale
        var sc = math.max(0, _range + tempLevel[0]) / _range;

        // Apply to rect-transforms.
        _bypassMeter.transform.localScale = new Vector3(sc.x, 1, 1);
        _lowPassMeter.transform.localScale = new Vector3(sc.y, 1, 1);
        _bandPassMeter.transform.localScale = new Vector3(sc.z, 1, 1);
        _highPassMeter.transform.localScale = new Vector3(sc.w, 1, 1);

        // Cleaning the temporaries up.
        tempFilter.Dispose();
        tempLevel.Dispose();

        samples.Dispose();
    }
}
