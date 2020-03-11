using System.Collections;
using System.Collections.Generic;
using System.IO;
using UnityEngine;
using UnityEditor;
using Unity.Collections;
using Unity.Jobs;
using Unity.Mathematics;

public class DFTPreprocessWindow : EditorWindow
{
    Object _clip = null;
    float _bandPassFreq = 670f;
    float _bandPassQ = 1.0f;
    MultibandFilter _filter;

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

    [MenuItem("Window/DFT Preprocess")]
    static void Init()
    {
        var window = EditorWindow.GetWindow<DFTPreprocessWindow>();
        window.Show();
    }

    void OnGUI()
    {
        GUILayout.Label("Base Settings", EditorStyles.boldLabel);
        _clip = EditorGUILayout.ObjectField("Audio Clip", _clip, typeof(AudioClip), false);
        _bandPassFreq = EditorGUILayout.Slider("Band Pass Center Frequency", _bandPassFreq, 50f, 2000f);
        _bandPassQ = EditorGUILayout.Slider("Band Pass !", _bandPassQ, 0.1f, 100f);

        if (GUILayout.Button("Start DFT Process"))
        {
            StartProcess();
        }
    }

    void StartProcess()
    {
        if (_clip == null)
        {
            Debug.LogError("Audio Clip is required");
            return;
        }

        var clip = _clip as AudioClip;

        string path = AssetDatabase.GetAssetPath(clip);
        Debug.Log($"Start audio prcessing: {path}");

        // Check errors
        if (!clip.LoadAudioData())
        {
            Debug.LogError("Failed to load audio");
            return;
        }

        if (clip.channels > 2)
        {
            Debug.LogError("3+ channels audioclip is not suppoerted.");
            return;
        }

        // Create buffer 
        var rawBuffer = new float[clip.samples * clip.channels];
        clip.GetData(rawBuffer, 0);

        if (clip.channels == 2)
        {
            // merge 2 channels
            var mergedBuffer = new float[clip.samples];
            for (int i = 0; i < mergedBuffer.Length; ++i)
            {
                mergedBuffer[i] = (rawBuffer[i * 2] + rawBuffer[i * 2 + 1]) * 0.5f;
            }
            rawBuffer = mergedBuffer;
        }

        const int WINDOW_WIDTH = 1024;
        int bufferLength = clip.samples % WINDOW_WIDTH == 0
            ? clip.samples
            : clip.samples + WINDOW_WIDTH - clip.samples % WINDOW_WIDTH;

        Debug.Log($"samples: {clip.samples} bufsamples: {bufferLength}");

        var buffer = new NativeArray<float>(bufferLength, Allocator.Persistent);
        NativeArray<float>.Copy(rawBuffer, buffer, rawBuffer.Length);

        var dftResult = new NativeArray<float4>(bufferLength / WINDOW_WIDTH, Allocator.Persistent);


        // Set filter parameter
        int sampleRate = clip.frequency;
        for (int i = 0; i < bufferLength; i += WINDOW_WIDTH)
        {
            var slice = new NativeSlice<float>(buffer, i, WINDOW_WIDTH);
            dftResult[i / WINDOW_WIDTH] = ExecuteDFT(ref slice, sampleRate);
        }

        string savePath = path.Replace(Path.GetExtension(path), ".bytes");
        SaveDFTFile(ref dftResult, savePath);
        Debug.Log($"Saved file: {savePath}");

        dftResult.Dispose();
        buffer.Dispose();
    }

    float4 ExecuteDFT(ref NativeSlice<float> slice, int sampleRate)
    {
        // Single element native array used to share structs with the job.
        var tempFilter = new NativeArray<MultibandFilter>
          (1, Allocator.TempJob, NativeArrayOptions.UninitializedMemory);

        var tempLevel = new NativeArray<float4>
          (1, Allocator.TempJob, NativeArrayOptions.UninitializedMemory);

        // Filter update
        _filter.SetParameter(_bandPassFreq / sampleRate, _bandPassQ);
        tempFilter[0] = _filter;

        // Run the job on the main thread.
        new FilterRmsJob
        {
            Input = slice,
            Filter = tempFilter,
            Output = tempLevel
        }.Run();

        // Preserve the filter state.
        _filter = tempFilter[0];

        float4 sc = tempLevel[0];

        tempFilter.Dispose();
        tempLevel.Dispose();

        return sc;
    }

    void SaveDFTFile(ref NativeArray<float4> data, string path)
    {
        string assetPath = Directory.GetParent(Application.dataPath).FullName;
        string fullpath = Path.Combine(assetPath, path);

        var bytes = data.ToRawBytes();
        Debug.Log($"Save to file: data: {data.Length} bytes: {bytes.Length}");
        File.WriteAllBytes(fullpath, bytes);
    }

}
