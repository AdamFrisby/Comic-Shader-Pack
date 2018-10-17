Shader "Custom/NPR Outline Shader" {
	Properties {
		_Background1("Background Bright Top", Color) = (1,1,1,1)
		_Background2("Background Bright Bottom", Color) = (1,1,1,1)
		_Background3("Background Dark Top", Color) = (1,1,1,1)
		_Background4("Background Dark Bottom", Color) = (1,1,1,1)
		_BackgroundAlphaTint("Background Alpha Tint", Color) = (1,0,1,1)
		_Line1("Line Bright Top", Color) = (1,1,1,1)
		_Line2("Line Bright Bottom", Color) = (1,1,1,1)
		_Line3("Line Dark Top", Color) = (1,1,1,1)
		_Line4("Line Dark Bottom", Color) = (1,1,1,1)

		_Near("Near Distance", float) = 0.1
		_Far("Far Distance", float) = 200

		_NearScale("Near Scale", float) = 0.0
		_FarScale("Far Scale", float) = 1.0

		_LineWidthModifier("Line Width Modifier", Range(0,1)) = 0.5
		_FWidthScale("F Width Scale", float) = 1.0


		_MainTex ("Line Map (R)", 2D) = "white" {}
		_Glossiness ("Smoothness", Range(0,1)) = 0.5
		_Metallic ("Metallic", Range(0,1)) = 0.0

	}
	SubShader {
		Tags { "RenderType"="Opaque" }
		LOD 200

		CGPROGRAM
		// Physically based Standard lighting model, and enable shadows on all light types
		#pragma surface surf StandardComic fullforwardshadows vertex:vert

		// Use shader model 3.0 target, to get nicer looking lighting
		#pragma target 4.0

		#define TAU 6.28318531
		#define PI 3.14159365
		const float3 wref = float3(1.0, 1.0, 1.0);

		float mod(float x, float y)
		{
			return x - y * floor(x / y);
		}

		float xyzF(float t) { return lerp(pow(t,1. / 3.), 7.787037*t + 0.139731, step(t,0.00885645)); }
		float xyzR(float t) { return lerp(t*t*t , 0.1284185*(t - 0.139731), step(t,0.20689655)); }
		float3 rgb2lch(in float3 c)
		{
			c = mul(float3x3(0.4124, 0.3576, 0.1805, 0.2126, 0.7152, 0.0722, 0.0193, 0.1192, 0.9505), c);
			c.x = xyzF(c.x / wref.x);
			c.y = xyzF(c.y / wref.y);
			c.z = xyzF(c.z / wref.z);
			float3 lab = float3(max(0.,116.0*c.y - 16.0), 500.0*(c.x - c.y), 200.0*(c.y - c.z));
			return float3(lab.x, length(float2(lab.y,lab.z)), atan2(lab.y, lab.z));
		}

		float3 lch2rgb(in float3 c)
		{
			c = float3(c.x, cos(c.z) * c.y, sin(c.z) * c.y);

			float lg = 1. / 116.*(c.x + 16.);
			float3 xyz = float3(wref.x*xyzR(lg + 0.002*c.y),
				wref.y*xyzR(lg),
				wref.z*xyzR(lg - 0.005*c.z));

			float3 rgb = mul(float3x3(3.2406, -1.5372,-0.4986,
				-0.9689,  1.8758, 0.0415,
				0.0557,  -0.2040, 1.0570), xyz);

			return rgb;
		}

		//cheaply lerp around a circle
		float lerpAng(in float a, in float b, in float x)
		{
			float ang = mod(mod((a - b), TAU) + PI*3., TAU) - PI;
			return ang*x + b;
		}

		//Linear interpolation between two colors in Lch space
		float3 lerpLch(in float3 a, in float3 b, in float x)
		{
			float hue = lerpAng(a.z, b.z, 1.0 - x);
			return float3(lerp(b.xy, a.xy, 1.0- x), hue);
		}

		float3 lerpCheap(in float3 a, in float3 b, in float x)
		{
			return lerp(a, b, smoothstep(0.0, 1.0, smoothstep(0.0, 1.0, x)));
		}

		float3 lerpGrad(in float3 a, in float3 b, in float x)
		{
			return lerpCheap(a, b, x);
		}


		// Lighting Model

		// Unity built-in shader source. Copyright (c) 2016 Unity Technologies. MIT license (see license.txt)


		// Main Physically Based BRDF
		// Derived from Disney work and based on Torrance-Sparrow micro-facet model
		//
		//   BRDF = kD / pi + kS * (D * V * F) / 4
		//   I = BRDF * NdotL
		//
		// * NDF (depending on UNITY_BRDF_GGX):
		//  a) Normalized BlinnPhong
		//  b) GGX
		// * Smith for Visiblity term
		// * Schlick approximation for Fresnel
			half4 BRDF1_Unity_PBS_Comic(half3 diffColor, half3 specColor, half oneMinusReflectivity, half smoothness,
				float3 normal, float3 viewDir,
				UnityLight light, UnityIndirect gi)
		{
			float perceptualRoughness = SmoothnessToPerceptualRoughness(smoothness);
			float3 halfDir = Unity_SafeNormalize(float3(light.dir) + viewDir);

			// NdotV should not be negative for visible pixels, but it can happen due to perspective projection and normal mapping
			// In this case normal should be modified to become valid (i.e facing camera) and not cause weird artifacts.
			// but this operation adds few ALU and users may not want it. Alternative is to simply take the abs of NdotV (less correct but works too).
			// Following define allow to control this. Set it to 0 if ALU is critical on your platform.
			// This correction is interesting for GGX with SmithJoint visibility function because artifacts are more visible in this case due to highlight edge of rough surface
			// Edit: Disable this code by default for now as it is not compatible with two sided lighting used in SpeedTree.
#define UNITY_HANDLE_CORRECTLY_NEGATIVE_NDOTV 0

#if UNITY_HANDLE_CORRECTLY_NEGATIVE_NDOTV
			// The amount we shift the normal toward the view vector is defined by the dot product.
			half shiftAmount = dot(normal, viewDir);
			normal = shiftAmount < 0.0f ? normal + viewDir * (-shiftAmount + 1e-5f) : normal;
			// A re-normalization should be applied here but as the shift is small we don't do it to save ALU.
			//normal = normalize(normal);

			half nv = saturate(dot(normal, viewDir)); // TODO: this saturate should no be necessary here
#else
			half nv = abs(dot(normal, viewDir));    // This abs allow to limit artifact
#endif

			half nl = saturate(dot(normal, light.dir));
			float nh = saturate(dot(normal, halfDir));

			half lv = saturate(dot(light.dir, viewDir));
			half lh = saturate(dot(light.dir, halfDir));

			// Diffuse term
			half diffuseTerm = DisneyDiffuse(nv, nl, lh, perceptualRoughness) * nl;

			// Specular term
			// HACK: theoretically we should divide diffuseTerm by Pi and not multiply specularTerm!
			// BUT 1) that will make shader look significantly darker than Legacy ones
			// and 2) on engine side "Non-important" lights have to be divided by Pi too in cases when they are injected into ambient SH
			float roughness = PerceptualRoughnessToRoughness(perceptualRoughness);
#if UNITY_BRDF_GGX
			// GGX with roughtness to 0 would mean no specular at all, using max(roughness, 0.002) here to match HDrenderloop roughtness remapping.
			roughness = max(roughness, 0.002);
			half V = SmithJointGGXVisibilityTerm(nl, nv, roughness);
			float D = GGXTerm(nh, roughness);
#else
			// Legacy
			half V = SmithBeckmannVisibilityTerm(nl, nv, roughness);
			half D = NDFBlinnPhongNormalizedTerm(nh, PerceptualRoughnessToSpecPower(perceptualRoughness));
#endif

			half specularTerm = V*D * UNITY_PI; // Torrance-Sparrow model, Fresnel is applied later

#   ifdef UNITY_COLORSPACE_GAMMA
			specularTerm = sqrt(max(1e-4h, specularTerm));
#   endif

			// specularTerm * nl can be NaN on Metal in some cases, use max() to make sure it's a sane value
			specularTerm = max(0, specularTerm * nl);
#if defined(_SPECULARHIGHLIGHTS_OFF)
			specularTerm = 0.0;
#endif

			// surfaceReduction = Int D(NdotH) * NdotH * Id(NdotL>0) dH = 1/(roughness^2+1)
			half surfaceReduction;
#   ifdef UNITY_COLORSPACE_GAMMA
			surfaceReduction = 1.0 - 0.28*roughness*perceptualRoughness;      // 1-0.28*x^3 as approximation for (1/(x^4+1))^(1/2.2) on the domain [0;1]
#   else
			surfaceReduction = 1.0 / (roughness*roughness + 1.0);           // fade \in [0.5;1]
#   endif

																			// To provide true Lambert lighting, we need to be able to kill specular completely.
			specularTerm *= any(specColor) ? 1.0 : 0.0;

			half grazingTerm = saturate(smoothness + (1 - oneMinusReflectivity));
			half3 color = diffColor * (gi.diffuse + light.color * diffuseTerm)
				+ specularTerm * light.color * FresnelTerm(specColor, lh)
				+ surfaceReduction * gi.specular * FresnelLerp(specColor, grazingTerm, nv);

			return half4(color, 1);
		}


		//-------------------------------------------------------------------------------------
		// Default BRDF to use:
#if !defined (UNITY_BRDF_PBS) // allow to explicitly override BRDF in custom shader
		// still add safe net for low shader models, otherwise we might end up with shaders failing to compile
#if SHADER_TARGET < 30
#define UNITY_BRDF_PBS BRDF3_Unity_PBS
#elif defined(UNITY_PBS_USE_BRDF3)
#define UNITY_BRDF_PBS BRDF3_Unity_PBS
#elif defined(UNITY_PBS_USE_BRDF2)
#define UNITY_BRDF_PBS BRDF2_Unity_PBS
#elif defined(UNITY_PBS_USE_BRDF1)
#define UNITY_BRDF_PBS BRDF1_Unity_PBS
#elif defined(SHADER_TARGET_SURFACE_ANALYSIS)
		// we do preprocess pass during shader analysis and we dont actually care about brdf as we need only inputs/outputs
#define UNITY_BRDF_PBS BRDF1_Unity_PBS
#else
#error something broke in auto-choosing BRDF
#endif
#endif

		struct SurfaceOutputStandardComic
		{
			fixed3 Albedo;      // base (diffuse or specular) color
			float3 Normal;      // tangent space normal, if written
			half3 Emission;
			fixed3 Background;
			half Metallic;      // 0=non-metal, 1=metal
								// Smoothness is the user facing name, it should be perceptual smoothness but user should not have to deal with it.
								// Everywhere in the code you meet smoothness it is perceptual smoothness
			half Smoothness;    // 0=rough, 1=smooth
			half Occlusion;     // occlusion (default 1)
			fixed Alpha;        // alpha for transparencies
		};

		inline half4 LightingStandardComic(SurfaceOutputStandardComic s, float3 viewDir, UnityGI gi)
		{
			s.Normal = normalize(s.Normal);

			half oneMinusReflectivity;
			half3 specColor;

			half3 realAlbedo = gi.light.color.r > 0.5 ? s.Albedo : s.Background;

			half3 backupAlbedo = realAlbedo;

			realAlbedo = DiffuseAndSpecularFromMetallic(realAlbedo, s.Metallic, /*out*/ specColor, /*out*/ oneMinusReflectivity);

			// shader relies on pre-multiply alpha-blend (_SrcBlend = One, _DstBlend = OneMinusSrcAlpha)
			// this is necessary to handle transparency in physically correct way - only diffuse component gets affected by alpha
			half outputAlpha;
			realAlbedo = PreMultiplyAlpha(realAlbedo, s.Alpha, oneMinusReflectivity, /*out*/ outputAlpha);

			half4 c = BRDF1_Unity_PBS_Comic(realAlbedo, specColor, oneMinusReflectivity, s.Smoothness, s.Normal, viewDir, gi.light, gi.indirect);

			c.rgb = min(c.rgb, backupAlbedo);

			c.a = outputAlpha;
			return c;
		}

		inline half4 LightingStandardComic_Deferred(SurfaceOutputStandardComic s, float3 viewDir, UnityGI gi, out half4 outGBuffer0, out half4 outGBuffer1, out half4 outGBuffer2)
		{
			half oneMinusReflectivity;
			half3 specColor;
			s.Albedo = DiffuseAndSpecularFromMetallic(s.Albedo, s.Metallic, /*out*/ specColor, /*out*/ oneMinusReflectivity);

			half4 c = UNITY_BRDF_PBS(s.Albedo, specColor, oneMinusReflectivity, s.Smoothness, s.Normal, viewDir, gi.light, gi.indirect);

			UnityStandardData data;
			data.diffuseColor = s.Albedo;
			data.occlusion = s.Occlusion;
			data.specularColor = specColor;
			data.smoothness = s.Smoothness;
			data.normalWorld = s.Normal;

			UnityStandardDataToGbuffer(data, outGBuffer0, outGBuffer1, outGBuffer2);

			half4 emission = half4(s.Emission + c.rgb, 1);
			return emission;
		}

		inline void LightingStandardComic_GI(
			SurfaceOutputStandardComic s,
			UnityGIInput data,
			inout UnityGI gi)
		{
#if defined(UNITY_PASS_DEFERRED) && UNITY_ENABLE_REFLECTION_BUFFERS
			gi = UnityGlobalIllumination(data, s.Occlusion, s.Normal);
#else
			Unity_GlossyEnvironmentData g = UnityGlossyEnvironmentSetup(s.Smoothness, data.worldViewDir, s.Normal, lerp(unity_ColorSpaceDielectricSpec.rgb, s.Albedo, s.Metallic));
			gi = UnityGlobalIllumination(data, s.Occlusion, s.Normal, g);
#endif
		}

		// End of Lighting Model


		sampler2D _MainTex;
		fixed4 _Background1;
		fixed4 _Background2;
		fixed4 _Background3;
		fixed4 _Background4;
		fixed4 _BackgroundAlphaTint;
		fixed4 _Line1;
		fixed4 _Line2;
		fixed4 _Line3;
		fixed4 _Line4;
		fixed _LineWidthModifier;
		fixed _FWidthScale;

		float _Near;
		float _Far;
		float _NearScale;
		float _FarScale;

		sampler2D_float _CameraDepthTexture;

		struct Input {
			float2 uv_MainTex;
			float4 screenPos;
			float3 worldPos;
			float eyeDepth;
		};

		half _Glossiness;
		half _Metallic;
		fixed4 _Color;

		// Add instancing support for this shader. You need to check 'Enable Instancing' on materials that use the shader.
		// See https://docs.unity3d.com/Manual/GPUInstancing.html for more information about instancing.
		// #pragma instancing_options assumeuniformscaling
		UNITY_INSTANCING_CBUFFER_START(Props)
			// put more per-instance properties here
		UNITY_INSTANCING_CBUFFER_END

		void vert (inout appdata_full v, out Input o)
		{
				UNITY_INITIALIZE_OUTPUT(Input, o);
				COMPUTE_EYEDEPTH(o.eyeDepth);
		}

		void surf (Input IN, inout SurfaceOutputStandardComic o) {

			float2 screenUV = IN.screenPos.xy / IN.screenPos.w;
			float y = saturate(IN.worldPos.y / 20);//screenUV.y;

			fixed3 colback = lerpGrad(_Background1.rgb, _Background2.rgb, y);
			fixed3 colline = lerpGrad(_Line1.rgb, _Line2.rgb, y);

			fixed3 darkback = lerpGrad(_Background3.rgb, _Background4.rgb, y);
			fixed3 darkline = lerpGrad(_Line3.rgb, _Line4.rgb, y);

			float rawZ = SAMPLE_DEPTH_TEXTURE_PROJ(_CameraDepthTexture, UNITY_PROJ_COORD(IN.screenPos));
			float sceneZ = LinearEyeDepth(rawZ);
			float partZ = IN.eyeDepth;

			float distance = saturate((partZ - _Near)/(_Far-_Near));


			// Albedo comes from a texture tinted by color
			fixed4 c = tex2D (_MainTex, IN.uv_MainTex);
			fixed modifiedLineValue = (c.r - lerp(_NearScale, _FarScale, distance));

			//modifiedLineValue = max(fwidth(modifiedLineValue) * _FWidthScale, 0.0001) - _LineWidthModifier;

			o.Albedo = lerpGrad(colback.rgb, colline.rgb, saturate((modifiedLineValue)*50)) * lerp(float3(1, 1, 1), _BackgroundAlphaTint.rgb, 1.0 - c.a);
			o.Background = lerpGrad(darkback.rgb, darkline.rgb, saturate((modifiedLineValue) * 50)) * lerp(float3(1,1,1), _BackgroundAlphaTint.rgb, 1.0 - c.a);
			// Metallic and smoothness come from slider variables
			o.Metallic = _Metallic;
			o.Smoothness = _Glossiness;
			o.Alpha = c.a;
		}
		ENDCG
	}
	FallBack "Diffuse"
}
