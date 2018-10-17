-- Gets our resource
resource = Space.GetResource("Image Effect");
material = resource.AsMaterial;

-- Creates our command buffer
 cbuf = Space.PostFX.CreateCommandBuffer();

-- "Blit"
 cbuf.Blit(material);

-- Applies it to the camera
 Space.PostFX.AddCommandBufferToCamera(cbuf, 18);
