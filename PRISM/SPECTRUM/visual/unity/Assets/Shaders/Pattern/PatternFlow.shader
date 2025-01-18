Shader "PRISM/PatternFlow"
{
    Properties
    {
        _MainTex ("Pattern Texture", 2D) = "white" {}
        _FlowIntensity ("Flow Intensity", Range(0,1)) = 0.5
        _MaterialResponse ("Material Response", Range(0,1)) = 0.5
        _RotationSpeed ("Rotation Speed", Range(0,2)) = 1.0
        _SpiralTightness ("Spiral Tightness", Range(0,5)) = 1.618
    }

    SubShader
    {
        Tags { "RenderType"="Transparent" "Queue"="Transparent" }
        LOD 100

        Pass
        {
            Blend SrcAlpha OneMinusSrcAlpha

            CGPROGRAM
            #pragma vertex vert
            #pragma fragment frag
            #include "UnityCG.cginc"

            // Define golden ratio constant
            #define PHI 1.618033988749895

            struct appdata
            {
                float4 vertex : POSITION;
                float2 uv : TEXCOORD0;
                float3 normal : NORMAL;
            };

            struct v2f
            {
                float2 uv : TEXCOORD0;
                float4 vertex : SV_POSITION;
                float3 worldPos : TEXCOORD1;
            };

            sampler2D _MainTex;
            float4 _MainTex_ST;
            float _FlowIntensity;
            float _MaterialResponse;
            float _RotationSpeed;
            float _SpiralTightness;

            v2f vert (appdata v)
            {
                v2f o;
                o.vertex = UnityObjectToClipPos(v.vertex);
                o.uv = TRANSFORM_TEX(v.uv, _MainTex);
                o.worldPos = mul(unity_ObjectToWorld, v.vertex).xyz;
                return o;
            }

            float2 polarCoord(float2 uv)
            {
                float2 delta = uv - float2(0.5, 0.5);
                float radius = length(delta) * 2;
                float angle = atan2(delta.y, delta.x);
                return float2(radius, angle);
            }

            fixed4 frag (v2f i) : SV_Target
            {
                // Convert to polar coordinates
                float2 polar = polarCoord(i.uv);
                float radius = polar.x;
                float angle = polar.y;

                // Create golden ratio based spirals
                float spiral1 = angle + _Time.y * (_RotationSpeed / PHI) + radius * (_SpiralTightness * PHI);
                float spiral2 = -angle + _Time.y * (_RotationSpeed * PHI) + radius * (_SpiralTightness / PHI);

                // Sample pattern data
                float4 patternData = tex2D(_MainTex, i.uv);

                // Create emergence and absence patterns with phi-based modulation
                float emergence = 0.5 + 0.5 * sin(spiral1 * PHI);
                float absence = 0.5 + 0.5 * sin(spiral2 / PHI);

                // Blend based on radius for smooth center using phi
                float centerBlend = smoothstep(0, 0.1 * PHI, radius);

                // Create final color
                float3 color = lerp(1, 0, absence) * emergence;
                color = lerp(1, color, centerBlend);

                // Apply pattern intensity
                float intensity = patternData.w * _FlowIntensity;
                color *= intensity;

                // Fade edges with phi-based radius
                float alpha = (1 - radius / PHI) * intensity;

                return float4(color, alpha);
            }
            ENDCG
        }
    }
}
