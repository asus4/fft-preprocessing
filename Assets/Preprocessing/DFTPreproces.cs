using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using Unity.Collections;
using Unity.Mathematics;


[RequireComponent(typeof(AudioSource))]
public class DFTPreproces : MonoBehaviour
{
    [SerializeField] TextAsset _dftFile = null;
    [SerializeField, Range(1f, 100f)] float _range = 60;
    [SerializeField] RectTransform _bypassMeter = null;
    [SerializeField] RectTransform _lowPassMeter = null;
    [SerializeField] RectTransform _bandPassMeter = null;
    [SerializeField] RectTransform _highPassMeter = null;

    AudioSource _source;
    NativeArray<float4> _dft;

    void Start()
    {
        _source = GetComponent<AudioSource>();
        _dft = NativeArrayExtension.FromRawBytes<float4>(_dftFile.bytes, Allocator.Persistent);
    }

    void Update()
    {
        int index = _source.timeSamples / 1000;

        if (index < 0 || index >= _dft.Length)
        {
            Debug.LogWarning($"index out of length: {index}");
            return;
        }

        var sc = math.max(0, _range + _dft[index]) / _range;

        // Apply to rect-transforms.
        _bypassMeter.transform.localScale = new Vector3(sc.x, 1, 1);
        _lowPassMeter.transform.localScale = new Vector3(sc.y, 1, 1);
        _bandPassMeter.transform.localScale = new Vector3(sc.z, 1, 1);
        _highPassMeter.transform.localScale = new Vector3(sc.w, 1, 1);
    }


}
