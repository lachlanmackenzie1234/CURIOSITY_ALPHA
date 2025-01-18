using UnityEngine;
#if UNITY_EDITOR
using UnityEditor;
#endif

namespace PRISM.SPECTRUM.Visual
{
    public class PatternSceneSetup : MonoBehaviour
    {
        [SerializeField] private ComputeShader patternCompute;
        private GameObject visualizationPlane;

        private void Awake()
        {
            // Load shader first
            if (patternCompute == null)
            {
#if UNITY_EDITOR
                patternCompute = AssetDatabase.LoadAssetAtPath<ComputeShader>(
                    "Assets/Shaders/Compute/PatternCompute.compute");
#endif

                if (patternCompute == null)
                {
                    Debug.LogError("[VERSION 2024.01-C] Could not load PatternCompute shader!");
                    return;
                }
            }

            Debug.Log($"[VERSION 2024.01-C] Loaded compute shader: {patternCompute.name}");
            SetupVisualizationPlane();
            SetupCamera();
        }

        private void SetupVisualizationPlane()
        {
            if (patternCompute == null)
            {
                Debug.LogError("[VERSION 2024.01-C] Cannot setup visualization - shader not loaded!");
                return;
            }

            // Create visualization plane
            visualizationPlane = new GameObject("VisualizationPlane");

            // Configure transform first
            visualizationPlane.transform.position = Vector3.zero;
            visualizationPlane.transform.rotation = Quaternion.identity;
            visualizationPlane.transform.localScale = new Vector3(4f, 4f, 1f);

            // Setup mesh components
            var meshFilter = visualizationPlane.AddComponent<MeshFilter>();
            var meshRenderer = visualizationPlane.AddComponent<MeshRenderer>();
            SetupMesh(meshFilter);

            // Add processor last and configure immediately
            var processor = visualizationPlane.AddComponent<PatternProcessor>();
            processor.enabled = false; // Disable until fully configured
            processor.patternCompute = patternCompute;
            var display = visualizationPlane.AddComponent<PatternDisplay>();
            processor.enabled = true; // Now enable
        }

        private void SetupMesh(MeshFilter meshFilter)
        {
            // Create circular mesh
            Mesh mesh = new Mesh();
            int segments = 32;
            Vector3[] vertices = new Vector3[segments + 1];
            Vector2[] uvs = new Vector2[segments + 1];
            int[] triangles = new int[segments * 3];

            vertices[0] = Vector3.zero;
            uvs[0] = new Vector2(0.5f, 0.5f);

            float angleStep = 360f / segments;
            for (int i = 0; i < segments; i++)
            {
                float angle = angleStep * i * Mathf.Deg2Rad;
                vertices[i + 1] = new Vector3(Mathf.Cos(angle), Mathf.Sin(angle), 0);
                uvs[i + 1] = new Vector2(0.5f + (Mathf.Cos(angle) * 0.5f), 0.5f + (Mathf.Sin(angle) * 0.5f));

                triangles[i * 3] = 0;
                triangles[i * 3 + 1] = i + 1;
                triangles[i * 3 + 2] = (i + 1) % segments + 1;
            }

            mesh.vertices = vertices;
            mesh.triangles = triangles;
            mesh.uv = uvs;
            mesh.RecalculateNormals();

            meshFilter.mesh = mesh;
        }

        private void SetupCamera()
        {
            var mainCamera = Camera.main;
            if (mainCamera == null)
            {
                var cameraObject = new GameObject("Main Camera");
                mainCamera = cameraObject.AddComponent<Camera>();
                mainCamera.tag = "MainCamera";
            }

            mainCamera.orthographic = true;
            mainCamera.orthographicSize = 3;
            mainCamera.transform.position = new Vector3(0, 0, -5);
            mainCamera.transform.LookAt(Vector3.zero);
        }
    }
}
