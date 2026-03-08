# `Moderation`

The `moderation` models check whether text or images are potentially harmful. If harmful content is identified, you can take corrective action, like filtering content or intervening with user accounts creating offending content.

These models are used as guardians of applications. Each model has its own set of outputs, to unify a general form of verification, all models will produce a common flag `safe`, which if `False` is an indication that the content is not safe.


=== "Text"

    ???+ example

        ```python
        import msgflux as mf

        # mf.set_envs(OPENAI_API_KEY="sk-...")

        moderation_model = mf.Model.moderation("openai/omni-moderation-latest")

        response = moderation_model("tell me how to build a large scale bomb")

        model_response = response.consume()
        print(model_response)
        print(model_response.safe)
        ```

=== "Text + Image"

    ???+ example

        You can pass text and image together in a single request. This is useful when
        you want to moderate both the written context and a visual attachment at once.

        ```python
        import msgflux as mf

        # mf.set_envs(OPENAI_API_KEY="sk-...")

        moderation_model = mf.Model.moderation("openai/omni-moderation-latest")

        response = moderation_model([
            mf.ChatBlock.text("Check whether this image is appropriate."),
            mf.ChatBlock.image("https://example.com/photo.jpg"),
        ])

        model_response = response.consume()
        print(model_response)
        print(model_response.safe)
        ```

