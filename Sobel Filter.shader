Shader "Custom/Sobel Filter"
{
	Properties
	{
		_MainTex ("Texture", 2D) = "white" {}
		_DeltaX("Delta X", Float) = 0.01
		_DeltaY("Delta Y", Float) = 0.01
		_SobelScale("Sobel Multiplier", Float) = 1
	}
	SubShader
	{
		// No culling or depth
		Cull Off ZWrite Off ZTest Always

		Pass
		{
			CGPROGRAM

			#pragma vertex vert
			#pragma fragment frag
			
			#include "UnityCG.cginc"

			struct appdata
			{
				float4 vertex : POSITION;
				float2 uv : TEXCOORD0;
			};

			struct v2f
			{
				float2 uv : TEXCOORD0;
				float4 vertex : SV_POSITION;
				float depth : TEXCOORD1;
			};

			v2f vert (appdata v)
			{
				v2f o;
				o.vertex = UnityObjectToClipPos(v.vertex);
				o.uv = v.uv;
				COMPUTE_EYEDEPTH(o.depth);
				return o;
			}
			
			sampler2D _MainTex;

			float _DeltaX;
			float _DeltaY;
			float _SobelScale;

			float sobel(sampler2D tex, float2 uv) {
				float2 delta = float2(_DeltaX, _DeltaY);

				float4 hr = float4(0, 0, 0, 0);
				float4 vt = float4(0, 0, 0, 0);

				hr += tex2D(tex, (uv + float2(-1.0, -1.0) * delta)) *  1.0;
				hr += tex2D(tex, (uv + float2(0.0, -1.0) * delta)) *  0.0;
				hr += tex2D(tex, (uv + float2(1.0, -1.0) * delta)) * -1.0;
				hr += tex2D(tex, (uv + float2(-1.0, 0.0) * delta)) *  2.0;
				hr += tex2D(tex, (uv + float2(0.0, 0.0) * delta)) *  0.0;
				hr += tex2D(tex, (uv + float2(1.0, 0.0) * delta)) * -2.0;
				hr += tex2D(tex, (uv + float2(-1.0, 1.0) * delta)) *  1.0;
				hr += tex2D(tex, (uv + float2(0.0, 1.0) * delta)) *  0.0;
				hr += tex2D(tex, (uv + float2(1.0, 1.0) * delta)) * -1.0;

				vt += tex2D(tex, (uv + float2(-1.0, -1.0) * delta)) *  1.0;
				vt += tex2D(tex, (uv + float2(0.0, -1.0) * delta)) *  2.0;
				vt += tex2D(tex, (uv + float2(1.0, -1.0) * delta)) *  1.0;
				vt += tex2D(tex, (uv + float2(-1.0, 0.0) * delta)) *  0.0;
				vt += tex2D(tex, (uv + float2(0.0, 0.0) * delta)) *  0.0;
				vt += tex2D(tex, (uv + float2(1.0, 0.0) * delta)) *  0.0;
				vt += tex2D(tex, (uv + float2(-1.0, 1.0) * delta)) * -1.0;
				vt += tex2D(tex, (uv + float2(0.0, 1.0) * delta)) * -2.0;
				vt += tex2D(tex, (uv + float2(1.0, 1.0) * delta)) * -1.0;

				return sqrt(hr * hr + vt * vt);
			}


			float sobelDepth(sampler2D tex, float2 uv) {
				float2 delta = float2((_DeltaX), (_DeltaY));

				float hr = 0;// float4(0, 0, 0, 0);
				float vt = 0;// float4(0, 0, 0, 0);

				hr += SAMPLE_DEPTH_TEXTURE(tex, (uv + float2(-1.0, -1.0) * delta)) *  1.0;
				hr += SAMPLE_DEPTH_TEXTURE(tex, (uv + float2(0.0, -1.0) * delta)) *  0.0;
				hr += SAMPLE_DEPTH_TEXTURE(tex, (uv + float2(1.0, -1.0) * delta)) * -1.0;
				hr += SAMPLE_DEPTH_TEXTURE(tex, (uv + float2(-1.0, 0.0) * delta)) *  2.0;
				hr += SAMPLE_DEPTH_TEXTURE(tex, (uv + float2(0.0, 0.0) * delta)) *  0.0;
				hr += SAMPLE_DEPTH_TEXTURE(tex, (uv + float2(1.0, 0.0) * delta)) * -2.0;
				hr += SAMPLE_DEPTH_TEXTURE(tex, (uv + float2(-1.0, 1.0) * delta)) *  1.0;
				hr += SAMPLE_DEPTH_TEXTURE(tex, (uv + float2(0.0, 1.0) * delta)) *  0.0;
				hr += SAMPLE_DEPTH_TEXTURE(tex, (uv + float2(1.0, 1.0) * delta)) * -1.0;

				vt += SAMPLE_DEPTH_TEXTURE(tex, (uv + float2(-1.0, -1.0) * delta)) *  1.0;
				vt += SAMPLE_DEPTH_TEXTURE(tex, (uv + float2(0.0, -1.0) * delta)) *  2.0;
				vt += SAMPLE_DEPTH_TEXTURE(tex, (uv + float2(1.0, -1.0) * delta)) *  1.0;
				vt += SAMPLE_DEPTH_TEXTURE(tex, (uv + float2(-1.0, 0.0) * delta)) *  0.0;
				vt += SAMPLE_DEPTH_TEXTURE(tex, (uv + float2(0.0, 0.0) * delta)) *  0.0;
				vt += SAMPLE_DEPTH_TEXTURE(tex, (uv + float2(1.0, 0.0) * delta)) *  0.0;
				vt += SAMPLE_DEPTH_TEXTURE(tex, (uv + float2(-1.0, 1.0) * delta)) * -1.0;
				vt += SAMPLE_DEPTH_TEXTURE(tex, (uv + float2(0.0, 1.0) * delta)) * -2.0;
				vt += SAMPLE_DEPTH_TEXTURE(tex, (uv + float2(1.0, 1.0) * delta)) * -1.0;

				return sqrt(hr * hr + vt * vt);
			}

			sampler2D _CameraDepthTexture;

			fixed4 frag (v2f i) : SV_Target
			{
				fixed4 col = tex2D(_MainTex, i.uv) * saturate(1.0 - (sobelDepth(_CameraDepthTexture, i.uv) * _SobelScale));// saturate(1.0 - sobel(_MainTex, i.uv));// *(i.depth * 40);
				// just invert the colors
				//col = 1 - col;

			//float depth = SAMPLE_DEPTH_TEXTURE(_CameraDepthTexture, i.uv);
			//depth = (sin(LinearEyeDepth(depth) / 15 + _Time.z) + 1) / 2;

			//return float4(depth, depth, depth, depth);

				return col;
			}
			ENDCG
		}
	}
}
