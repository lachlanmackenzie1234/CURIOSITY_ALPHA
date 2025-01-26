using UnityEngine;

namespace PRISM.SPECTRUM.Visual
{
    [RequireComponent(typeof(MeshRenderer))]
    public class PatternDisplay : MonoBehaviour
    {
        [SerializeField] private PatternProcessor patternProcessor;

        [Header("Material Response")]
        [SerializeField, Range(0, 1)] private float flowIntensity = 0.5f;
        [SerializeField, Range(0, 1)] private float materialResponse = 0.5f;

        private Material patternMaterial;
        private static readonly int MainTex = Shader.PropertyToID("_MainTex");
        private static readonly int FlowIntensity = Shader.PropertyToID("_FlowIntensity");
        private static readonly int MaterialResponse = Shader.PropertyToID("_MaterialResponse");

        private void Start()
        {
            // Create material instance
            var renderer = GetComponent<MeshRenderer>();
            patternMaterial = new Material(Shader.Find("PRISM/PatternFlow"));
            renderer.material = patternMaterial;

            // Initial parameter setup
            UpdateMaterialParameters();
        }

        private void Update()
        {
            if (patternProcessor != null)
            {
                // Connect compute output to material
                patternMaterial.SetTexture(MainTex, patternProcessor.GetResultTexture());

                // Update response parameters
                UpdateMaterialParameters();
            }
        }

        private void UpdateMaterialParameters()
        {
            patternMaterial.SetFloat(FlowIntensity, flowIntensity);
            patternMaterial.SetFloat(MaterialResponse, materialResponse);
        }

        private void OnDestroy()
        {
            if (patternMaterial != null)
            {
                if (Application.isPlaying)
                    Destroy(patternMaterial);
                else
                    DestroyImmediate(patternMaterial);
            }
        }
    }
}
