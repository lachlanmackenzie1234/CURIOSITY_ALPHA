using UnityEngine;
using Unity.Profiling;
using System.Collections;
#if UNITY_EDITOR
using UnityEditor;
#endif

namespace PRISM.SPECTRUM.Visual
{
    [RequireComponent(typeof(MeshRenderer))]
    public class PatternProcessor : MonoBehaviour
    {
        [SerializeField] private ComputeShader _patternCompute;
        public ComputeShader patternCompute
        {
            get => _patternCompute;
            set
            {
                if (value != _patternCompute)
                {
                    _patternCompute = value;
                    Debug.Log($"[VERSION 2024.01-D] PatternCompute shader assigned: {(_patternCompute != null ? _patternCompute.name : "NULL")}");
                    TryInitialize();
                }
            }
        }

        // Performance monitoring
        private static readonly ProfilerMarker _performanceMarker =
            new ProfilerMarker("PatternProcessor.Process");

        // Compute resources
        private ComputeBuffer patternBuffer;
        private RenderTexture resultTexture;
        private int kernelIndex;
        private bool isInitialized = false;

        // Pattern data
        private struct PatternData
        {
            public Vector3 position;
            public float intensity;
            public float phase;
            public float frequency;
        }

        private void TryInitialize()
        {
            if (!enabled || isInitialized || _patternCompute == null) return;

            Debug.Log($"[VERSION 2024.01-D] Attempting initialization for {gameObject.name}");
            InitializeResources();
        }

        private void Awake()
        {
            Debug.Log($"[VERSION 2024.01-D] PatternProcessor.Awake on {gameObject.name}");
        }

        private void OnEnable()
        {
            Debug.Log($"[VERSION 2024.01-D] PatternProcessor.OnEnable on {gameObject.name}");
            TryInitialize();
        }

        private void Start()
        {
            Debug.Log($"[VERSION 2024.01-D] PatternProcessor.Start on {gameObject.name}");
            TryInitialize();
        }

        private void InitializeResources()
        {
            if (isInitialized)
            {
                Debug.Log("[VERSION 2024.01-D] Already initialized");
                return;
            }

            if (_patternCompute == null)
            {
                Debug.LogError($"[VERSION 2024.01-D] Cannot initialize - no compute shader assigned to {gameObject.name}");
                return;
            }

            Debug.Log($"[VERSION 2024.01-D] Initializing with shader: {_patternCompute.name}");

            try
            {
                kernelIndex = _patternCompute.FindKernel("ProcessPatterns");
                InitializeRenderTexture();
                SetupPatternBuffer();
                isInitialized = true;
                Debug.Log("[VERSION 2024.01-D] Initialization complete");
            }
            catch (System.Exception e)
            {
                Debug.LogError($"[VERSION 2024.01-D] Initialization failed: {e.Message}\n{e.StackTrace}");
                ReleaseResources();
            }
        }

        private void InitializeRenderTexture()
        {
            // Release old texture if it exists
            if (resultTexture != null)
            {
                resultTexture.Release();
                resultTexture = null;
            }

            // Create render texture for compute shader output
            resultTexture = new RenderTexture(Screen.width, Screen.height, 0);
            resultTexture.enableRandomWrite = true;
            resultTexture.Create();

            // Set texture in compute shader
            patternCompute.SetTexture(kernelIndex, "Result", resultTexture);
        }

        private void SetupPatternBuffer()
        {
            // Release old buffer if it exists
            if (patternBuffer != null)
            {
                patternBuffer.Release();
                patternBuffer = null;
            }

            // Example pattern data
            PatternData[] patterns = new PatternData[1];
            patterns[0] = new PatternData
            {
                position = Vector3.zero,
                intensity = 1.0f,
                phase = 0.0f,
                frequency = 1.0f
            };

            // Create and set buffer
            patternBuffer = new ComputeBuffer(patterns.Length, sizeof(float) * 6);
            patternBuffer.SetData(patterns);
            patternCompute.SetBuffer(kernelIndex, "patterns", patternBuffer);
            patternCompute.SetInt("patternCount", patterns.Length);
        }

        private void Update()
        {
            if (!isInitialized || patternCompute == null) return;

            using (_performanceMarker.Auto())
            {
                // Update parameters
                patternCompute.SetFloat("deltaTime", Time.deltaTime);
                patternCompute.SetVector("dimensions",
                    new Vector3(Screen.width, Screen.height, 0));

                // Dispatch compute shader
                int threadGroupsX = Mathf.CeilToInt(Screen.width / 8.0f);
                int threadGroupsY = Mathf.CeilToInt(Screen.height / 8.0f);
                patternCompute.Dispatch(kernelIndex, threadGroupsX, threadGroupsY, 1);
            }
        }

        private void ReleaseResources()
        {
            if (patternBuffer != null)
            {
                patternBuffer.Release();
                patternBuffer = null;
            }

            if (resultTexture != null)
            {
                resultTexture.Release();
                resultTexture = null;
            }

            isInitialized = false;
            Debug.Log("PatternProcessor: Resources released");
        }

        private void OnDisable()
        {
            if (isInitialized)
            {
                Debug.Log("[VERSION 2024.01-D] Releasing resources due to disable");
                ReleaseResources();
            }
        }

        private void OnDestroy()
        {
            if (isInitialized)
            {
                Debug.Log("[VERSION 2024.01-D] Releasing resources due to destroy");
                ReleaseResources();
            }
        }

        // Public interface for pattern updates
        public RenderTexture GetResultTexture()
        {
            return resultTexture;
        }
    }
}
